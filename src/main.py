# Projet H.1 — ECE | Antoine BALSSA - Antoine BOURDELOIS - Augustin COUET

# Architecture :
#   - Universe Selection : top 50 actions US par market cap (dynamique)
#   - Alpha Model : ensemble RF + Gradient Boosting (23 features)
#   - Portfolio : position sizing proportionnel à la confiance ML
#   - Risk Management : stop-loss -5% par position (LEAN natif)
#   - Validation : Walk-Forward OOS + robustesse sur 6 périodes de 5 ans

# RÉSULTATS :
#   - Sharpe moyen : 0.553 sur 6 périodes (2010-2024)
#   - 5/6 périodes battent le SPY
#   - lookback 378j validé comme optimal (on a testé par rapport à 252)

from AlgorithmImports import *
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score


# CLASSE SymbolData
# Stocke les indicateurs techniques et calcule les 23 features pour chaque action
# Doit être inline dans main.py (règle QC : AlgorithmImports non importable ailleurs)

class SymbolData:
    """
    Gère les indicateurs et features d'une action individuelle.
    Instanciée dans OnSecuritiesChanged() pour chaque action ajoutée à l'univers.
    """

    def __init__(self, algorithm, symbol):
        self.symbol    = symbol
        self.algorithm = algorithm

        # --- Indicateurs techniques (calculés automatiquement par LEAN) ---
        # SMA : moyennes mobiles pour détecter la tendance
        self.sma_20  = algorithm.SMA(symbol, 20,  Resolution.Daily)
        self.sma_50  = algorithm.SMA(symbol, 50,  Resolution.Daily)
        self.sma_200 = algorithm.SMA(symbol, 200, Resolution.Daily)  # tendance long terme

        # RSI : relative strength index, mesure la sur-achat/sur-vente (0-100)
        self.rsi  = algorithm.RSI(symbol, 14, MovingAverageType.Wilders, Resolution.Daily)

        # MACD : momentum via croisement de 2 EMA (12 et 26 jours)
        self.macd = algorithm.MACD(symbol, 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily)

        # Bollinger Bands : enveloppe de volatilité autour de la SMA20
        self.bb   = algorithm.BB(symbol, 20, 2, MovingAverageType.Simple, Resolution.Daily)

        # ADX : Average Directional Index, force de la tendance (0=pas de tendance, 100=tendance forte)
        self.adx  = algorithm.ADX(symbol, 14, Resolution.Daily)

        # ATR : Average True Range, mesure la volatilité absolue en $
        self.atr  = algorithm.ATR(symbol, 14, MovingAverageType.Simple, Resolution.Daily)

        # Stochastique : compare le prix de clôture à son range récent (0-100)
        self.stoch = algorithm.STO(symbol, 14, 3, 3, Resolution.Daily)

        # --- Fenêtres glissantes pour les calculs manuels ---
        # maxlen=504 = 2 ans de données (buffer pour lookback 378j + marge)
        self.close_window  = deque(maxlen=504)  # prix de clôture historiques
        self.volume_window = deque(maxlen=504)  # volumes historiques

        # --- Données fondamentales (mises à jour mensellement via FineFilter) ---
        self.pe_ratio    = None  # Price/Earnings ratio
        self.pb_ratio    = None  # Price/Book ratio
        self.roe         = None  # Return on Equity
        self.debt_equity = None  # Ratio dette/capitaux propres

        # Flag : True quand l'action a accumulé assez de données pour être tradée
        self.is_ready = False

    def Update(self, trade_bar):
        """
        Appelé à chaque nouveau bar quotidien via OnData().
        Stocke le prix et le volume, puis vérifie si l'action est prête.
        is_ready = True quand on a 378 jours de données ET tous les indicateurs prêts.
        378 jours = notre lookback_days → garantit que GetTrainingData() a assez d'historique.
        """
        self.close_window.append(float(trade_bar.Close))
        self.volume_window.append(float(trade_bar.Volume))
        self.is_ready = (
            len(self.close_window) >= 378     # 378j minimum = lookback_days
            and self.sma_20.IsReady           # 20j pour SMA20
            and self.sma_50.IsReady           # 50j pour SMA50
            and self.sma_200.IsReady          # 200j pour SMA200
            and self.rsi.IsReady              # 14j pour RSI
            and self.macd.IsReady             # 26j pour MACD
            and self.bb.IsReady               # 20j pour BB
            and self.adx.IsReady              # 14j pour ADX
            and self.atr.IsReady              # 14j pour ATR
            and self.stoch.IsReady            # 14j pour Stochastique
        )

    def UpdateFundamentals(self, fundamental):
        """
        Appelé dans FineFilter() lors de la sélection mensuelle d'univers.
        Met à jour les ratios fondamentaux (données Morningstar point-in-time).
        Les valeurs invalides ou négatives sont ignorées (→ restent None → 0.0 dans les features).
        """
        try:
            if fundamental.ValuationRatios.PERatio > 0:
                self.pe_ratio = float(fundamental.ValuationRatios.PERatio)
            if fundamental.ValuationRatios.PBRatio > 0:
                self.pb_ratio = float(fundamental.ValuationRatios.PBRatio)
            if fundamental.OperationRatios.ROE.Value != 0:
                self.roe = float(fundamental.OperationRatios.ROE.Value)
            if fundamental.OperationRatios.TotalDebtEquityRatio.Value != 0:
                self.debt_equity = float(fundamental.OperationRatios.TotalDebtEquityRatio.Value)
        except:
            pass  # Données manquantes → on garde les valeurs précédentes

    def GetFeatures(self):
        """
        Calcule et retourne le vecteur de 23 features normalisées.
        Retourne None si l'action n'est pas encore prête.

        Les 23 features sont regroupées en 5 catégories :
        1. Returns multi-timeframe (4 features) : signal momentum
        2. Volatilité et ATR (2 features) : mesure du risque
        3. Indicateurs techniques (8 features) : RSI, MACD, BB, MA ratio, ADX, Stoch
        4. Multi-timeframe momentum (3 features) : momentum 60j, vol 60j, MA ratio 50/200
        5. Volume (2 features) : volume ratio et tendance
        6. Fondamentaux (4 features) : PE, PB, ROE, D/E
        """
        if not self.is_ready:
            return None

        prices  = list(self.close_window)
        volumes = list(self.volume_window)
        cp = prices[-1]  # prix actuel (close du jour)
        f  = {}

        #  Catégorie 1 : Returns multi-timeframe 
        # Mesure le momentum sur différentes périodes
        f['return_1d']  = (prices[-1] - prices[-2])  / prices[-2]   # rendement 1 jour
        f['return_5d']  = (prices[-1] - prices[-6])  / prices[-6]   # rendement 1 semaine
        f['return_10d'] = (prices[-1] - prices[-11]) / prices[-11]  # rendement 2 semaines
        f['return_20d'] = (prices[-1] - prices[-21]) / prices[-21]  # rendement 1 mois

        # ── Catégorie 2 : Volatilité 
        r20 = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(len(prices)-20, len(prices))]
        f['volatility_20d'] = float(np.std(r20))   # volatilité annualisable sur 20j
        f['atr_norm']       = self.atr.Current.Value / cp  # ATR normalisé par le prix

        # ── Catégorie 3 : Indicateurs techniques 
        # RSI normalisé : (RSI - 50) / 50 → [-1, 1] (0 = neutre, +1 = survente, -1 = surachat)
        f['rsi_norm']       = (self.rsi.Current.Value - 50) / 50

        # MACD normalisé par le prix (valeur absolue peu comparable entre actions)
        f['macd_norm']      = self.macd.Current.Value / cp
        f['macd_hist_norm'] = (self.macd.Current.Value - self.macd.Signal.Current.Value) / cp

        # Bollinger Bands : position du prix dans la bande (0=bas, 1=haut, 0.5=milieu)
        bb_u = self.bb.UpperBand.Current.Value
        bb_l = self.bb.LowerBand.Current.Value
        bb_r = bb_u - bb_l
        f['bb_percent'] = (cp - bb_l) / bb_r if bb_r > 0 else 0.5  # %B
        f['bb_width']   = bb_r / self.bb.MiddleBand.Current.Value   # largeur relative

        # Ratio SMA20/SMA50 : >1 = tendance haussière court terme
        f['ma_ratio_20_50'] = self.sma_20.Current.Value / self.sma_50.Current.Value

        # ADX normalisé : [0,1] (>0.25 = tendance significative)
        f['adx_norm']  = self.adx.Current.Value / 100

        # Stochastique normalisé : (K - 50) / 50 → [-1, 1]
        f['stoch_norm'] = (self.stoch.StochK.Current.Value - 50) / 50

        #  Catégorie 4 : Multi-timeframe (long terme) 
        f['momentum_60d'] = (prices[-1] - prices[-61]) / prices[-61]  # momentum 3 mois

        r60 = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(len(prices)-60, len(prices))]
        f['volatility_60d'] = float(np.std(r60))  # volatilité sur 3 mois

        # Ratio SMA50/SMA200 : >1 = tendance haussière long terme (golden cross)
        f['ma_ratio_50_200'] = self.sma_50.Current.Value / self.sma_200.Current.Value

        #  Catégorie 5 : Volume
        av20 = np.mean(volumes[-20:])  # volume moyen 20j
        f['volume_ratio'] = volumes[-1] / av20 if av20 > 0 else 1.0        # ratio volume/moyenne
        f['volume_trend'] = np.mean(volumes[-5:]) / av20 if av20 > 0 else 1.0  # tendance 5j

        #  Catégorie 6 : Fondamentaux 
        # Normalisations logarithmiques pour compresser les distributions asymétriques
        f['pe_norm']          = np.log(self.pe_ratio) / np.log(50) if self.pe_ratio and 0 < self.pe_ratio < 200 else 0.0
        f['pb_norm']          = np.log(self.pb_ratio + 1) / np.log(20) if self.pb_ratio and 0 < self.pb_ratio < 50 else 0.0
        f['roe_norm']         = max(-0.5, min(0.5, self.roe)) if self.roe is not None else 0.0  # clamping [-0.5, 0.5]
        f['debt_equity_norm'] = min(2.0, self.debt_equity) / 2.0 if self.debt_equity is not None and self.debt_equity >= 0 else 0.0

        return f


# ALGORITHME PRINCIPAL

class MLAlphaStrategy(QCAlgorithm):

    def Initialize(self):
        #   PARAMÈTRES DE DATE — À MODIFIER POUR CHANGER LA PÉRIODE 
        # Règle empirique : SetStartDate + 18 mois ≈ premier trade
        # Exemples pour avoir 5 ans de trading réel :
        #
        #   Premier trade 2010 → SetStartDate(2008, 6, 1) / SetEndDate(2014, 12, 31)
        #   Premier trade 2012 → SetStartDate(2010, 6, 1) / SetEndDate(2016, 12, 31)
        #   Premier trade 2014 → SetStartDate(2012, 5, 1) / SetEndDate(2018, 12, 31)
        #   Premier trade 2016 → SetStartDate(2014, 6, 1) / SetEndDate(2020, 12, 31)
        #   Premier trade 2018 → SetStartDate(2016, 6, 1) / SetEndDate(2022, 12, 31)
        #   Premier trade 2020 → SetStartDate(2018, 4, 1) / SetEndDate(2024, 12, 31)

        #attention erreur bizarre quand je test pour 2010 et 2014 il manque des données donc ça lance que en avril de la même année. 
        # ça fait comme s'il y a avait un manque de données
        
        #  POUR DÉMONSTRATION : MODIFIER UNIQUEMENT CES 2 LIGNES CI-DESSOUS 
        self.SetStartDate(2018, 4, 1)   # MODIFIER : mois X, 18 mois avant le premier trade
        self.SetEndDate(2024, 12, 31)   # MODIFIER : dernier jour de la période (= start_trade + 5 ans)

        self.SetCash(100000)          # capital initial : 100 000$
        self.SetBenchmark("SPY")      # benchmark : S&P 500 buy-and-hold

        #  Paramètres Universe Selection 
        self.num_stocks        = 50    # top 50 actions par market cap
        self.min_price         = 10.0  # filtre : prix > $10 (évite les penny stocks)
        self.min_dollar_volume = 5e6   # filtre : volume dollar > $5M/jour (liquidité)
        self.min_market_cap    = 2e9   # filtre : capitalisation > $2Mds

        #  Paramètres ML 
        self.lookback_days     = 378   # fenêtre d'entraînement : 378j ≈ 1.5 an
                                       # validé comme optimal vs 252j sur 6 périodes
        self.prediction_horizon = 5    # on prédit si le prix monte dans 5 jours
        self.probability_threshold = 0.60  # seuil de confiance minimum pour trader

        #  Paramètres Portfolio 
        self.max_position_weight = 0.10  # max 10% du portfolio par action
        self.max_total_invested  = 0.90  # max 90% investi (10% cash de sécurité)

        #  Paramètres Walk-Forward Validation 
        self.wf_test  = 21   # 21 jours de test OOS (1 mois)
        self.wf_train = 378  # 378 jours d'entraînement (= lookback_days)

        # Noms des 23 features dans l'ordre fixe (doit correspondre à GetFeatures)
        self.feature_names = [
            'return_1d', 'return_5d', 'return_10d', 'return_20d',  # returns
            'volatility_20d', 'atr_norm',                           # volatilité
            'rsi_norm', 'macd_norm', 'macd_hist_norm',              # oscillateurs
            'bb_percent', 'bb_width',                               # Bollinger
            'ma_ratio_20_50', 'adx_norm', 'stoch_norm',            # trend/momentum
            'momentum_60d', 'volatility_60d', 'ma_ratio_50_200',   # long terme
            'volume_ratio', 'volume_trend',                         # volume
            'pe_norm', 'pb_norm', 'roe_norm', 'debt_equity_norm'   # fondamentaux
        ]

        #  Configuration LEAN 
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseFilter, self.FineFilter)
        self.SetExecution(ImmediateExecutionModel())  # exécution au prix du marché

        # Stop-loss natif LEAN : liquide automatiquement si une position perd -5%
        # Plus réactif qu'un trailing stop manuel
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.05))

        # Warmup : charge les données historiques avant de commencer
        # LEAN remonte dans le passé pour remplir les indicateurs
        self.SetWarmup(self.lookback_days + 60)

        #  État interne 
        self.symbol_data         = {}     # {Symbol: SymbolData}
        self.rf_model            = None   # Random Forest (retrain mensuel)
        self.gb_model            = None   # Gradient Boosting (retrain mensuel)
        self.scaler              = None   # StandardScaler (retrain mensuel)
        self.model_trained       = False  # True dès le premier retrain réussi
        self.train_count         = 0      # compteur de retrains
        self.last_train_samples  = 0

        # Métriques Walk-Forward OOS (accumulées sur toute la période)
        self.wf_accs  = []  # accuracy par fold
        self.wf_precs = []  # precision par fold
        self.wf_f1s   = []  # F1-score par fold

        # Trade Log : suivi win rate par niveau de confiance ML
        # Buckets : 0.60-0.62 / 0.62-0.65 / 0.65-0.68 / 0.68-0.70 / 0.70-0.75 / 0.75+
        self.proba_buckets = {
            '0.60-0.62': {'wins': 0, 'total': 0, 'pnl': 0.0},
            '0.62-0.65': {'wins': 0, 'total': 0, 'pnl': 0.0},
            '0.65-0.68': {'wins': 0, 'total': 0, 'pnl': 0.0},
            '0.68-0.70': {'wins': 0, 'total': 0, 'pnl': 0.0},
            '0.70-0.75': {'wins': 0, 'total': 0, 'pnl': 0.0},
            '0.75+':     {'wins': 0, 'total': 0, 'pnl': 0.0},
        }
        # Positions ouvertes : {Symbol: {'proba': float, 'entry_price': float}}
        self.open_trade_probas = {}

        # Scheduling 
        # Retrain complet le 1er jour de chaque mois à 10h00
        self.Schedule.On(
            self.DateRules.MonthStart(),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.RetrainModel
        )
        # Génération des signaux chaque jour à 10h30
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.AfterMarketOpen("SPY", 60),
            self.GenerateSignals
        )

        self.Log("=" * 60)
        self.Log("ML ALPHA v8 FINALE — ECE School 2026")
        self.Log(f"  Période  : {self.StartDate.date()} → {self.EndDate.date()}")
        self.Log(f"  Stocks   : top {self.num_stocks} US par market cap")
        self.Log(f"  Modèles  : RF(200 arbres) + GB(150 arbres) — ensemble")
        self.Log(f"  Lookback : {self.lookback_days}j | Horizon : {self.prediction_horizon}j")
        self.Log(f"  Seuil    : proba > {self.probability_threshold}")
        self.Log(f"  Features : {len(self.feature_names)} (techniques + fondamentaux)")
        self.Log("=" * 60)

    # UNIVERSE SELECTION

    def CoarseFilter(self, coarse):
        """
        Filtre grossier sur ~10 000 actions US (données quotidiennes).
        Garde les 150 plus liquides avec données fondamentales disponibles.
        DollarVolume et prix sont des données point-in-time → pas de biais futur.
        """
        if self.IsWarmingUp:
            return Universe.Unchanged  # ne pas changer l'univers pendant le warmup

        filtered = [
            x for x in coarse
            if x.HasFundamentalData          # données fondamentales Morningstar disponibles
            and x.Price > self.min_price     # prix > $10
            and x.DollarVolume > self.min_dollar_volume  # liquidité suffisante
        ]
        # Top 150 par volume dollar → pré-sélection pour FineFilter
        return [x.Symbol for x in sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)[:150]]

    def FineFilter(self, fine):
        """
        Filtre fin sur les 150 pré-sélectionnés.
        Utilise les données fondamentales Morningstar (MarketCap point-in-time).
        Garde les 50 plus grandes capitalisations.
        Met à jour les fondamentaux des SymbolData existants.
        """
        filtered = [x for x in fine if x.MarketCap > self.min_market_cap]
        top50    = sorted(filtered, key=lambda x: x.MarketCap, reverse=True)[:self.num_stocks]

        # Mise à jour des fondamentaux pour chaque action sélectionnée
        for f in top50:
            if f.Symbol in self.symbol_data:
                self.symbol_data[f.Symbol].UpdateFundamentals(f)

        return [x.Symbol for x in top50]

    # GESTION DES DONNÉES

    def OnData(self, data):
        """
        Appelé à chaque bar quotidien.
        Met à jour les fenêtres de prix/volume pour chaque action de l'univers.
        """
        if self.IsWarmingUp:
            return

        for symbol, sd in self.symbol_data.items():
            if data.Bars.ContainsKey(symbol):
                sd.Update(data.Bars[symbol])

    def OnSecuritiesChanged(self, changes):
        """
        Appelé quand l'univers change (ajout/suppression d'actions).
        Crée ou supprime les SymbolData correspondants.
        """
        # Nouvelles actions → créer leur SymbolData
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = SymbolData(self, symbol)

        # Actions retirées → nettoyer
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.symbol_data:
                del self.symbol_data[symbol]
            self.open_trade_probas.pop(symbol, None)

    # TRADE LOG — OnOrderEvent

    def OnOrderEvent(self, orderEvent):
        """
        Appelé automatiquement par LEAN à chaque ordre exécuté.
        Capture TOUTES les fermetures de position :
          - Signal SELL du modèle ML
          - Stop-loss -5% déclenché par MaximumDrawdownPercentPerSecurity
          - Liquidation forcée (sortie d'univers)
        
        Avantage vs _CloseTradeLog() manuel : les stops LEAN ne passaient pas
        par notre fonction → les trades perdants n'étaient pas comptés → biais.
        """
        if self.IsWarmingUp: return
        if orderEvent.Status != OrderStatus.Filled: return

        symbol = orderEvent.Symbol
        qty    = orderEvent.FillQuantity

        # On s'intéresse uniquement aux ventes (fermetures de position longue)
        if qty >= 0: return
        if symbol not in self.open_trade_probas: return

        # Calcul du résultat du trade
        info   = self.open_trade_probas.pop(symbol)
        proba  = info['proba']
        entry  = info['entry_price']
        exit_p = orderEvent.FillPrice
        if entry <= 0 or exit_p <= 0: return

        ret    = (exit_p - entry) / entry  # rendement du trade
        is_win = 1 if ret > 0 else 0

        # Affectation au bucket de confiance correspondant
        if   proba < 0.62: bk = '0.60-0.62'
        elif proba < 0.65: bk = '0.62-0.65'
        elif proba < 0.68: bk = '0.65-0.68'
        elif proba < 0.70: bk = '0.68-0.70'
        elif proba < 0.75: bk = '0.70-0.75'
        else:              bk = '0.75+'

        self.proba_buckets[bk]['wins']  += is_win
        self.proba_buckets[bk]['total'] += 1
        self.proba_buckets[bk]['pnl']   += ret

    # RETRAIN MENSUEL — Walk-Forward Validation

    def OnWarmupFinished(self):
        """
        Déclenché automatiquement par LEAN à la fin du warmup.
        Force un premier retrain immédiat pour que le modèle soit prêt
        dès le premier jour de trading réel.
        """
        self.RetrainModel()

    def RetrainModel(self):
        """
        Retrain complet des modèles ML.
        Exécuté le 1er de chaque mois + immédiatement après le warmup.

        Pipeline :
        1. Collecte des données historiques (lookback_days jours par action)
        2. Walk-Forward OOS : évaluation honnête sur données non vues
        3. Retrain RF + GB sur toutes les données disponibles
        4. Log des top-5 features (importance RF)
        """
        if self.IsWarmingUp: return

        X, y, _, _ = self.GetTrainingData()
        if X is None or len(X) < 200:
            return

        n = len(X)

        #  Walk-Forward OOS 
        # Entraîne sur [ts:te] et évalue sur [te:], jamais vu pendant l'entraînement
        if n > self.wf_train + self.wf_test:
            te  = n - self.wf_test
            ts  = max(0, te - self.wf_train)
            sc  = StandardScaler()
            Xtr = sc.fit_transform(X[ts:te])
            Xts = sc.transform(X[te:])
            yts = y[te:]

            rf_wf = RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=10,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            rf_wf.fit(Xtr, y[ts:te])
            yp = rf_wf.predict(Xts)

            oa = accuracy_score(yts, yp)
            op = precision_score(yts, yp, zero_division=0)
            of = f1_score(yts, yp, zero_division=0)
            self.wf_accs.append(oa); self.wf_precs.append(op); self.wf_f1s.append(of)
            self.Debug(f"  [WF-OOS] acc={oa:.3f} prec={op:.3f} f1={of:.3f} | moy_acc={np.mean(self.wf_accs):.3f}")

        #  Retrain final sur toutes les données 
        self.scaler  = StandardScaler()
        Xsc = self.scaler.fit_transform(X)

        # Random Forest : robuste, peu d'overfitting, parallélisable
        self.rf_model = RandomForestClassifier(
            n_estimators=200,      # 200 arbres de décision
            max_depth=5,           # profondeur limitée → anti-surapprentissage
            min_samples_leaf=10,   # minimum 10 échantillons par feuille
            max_features='sqrt',   # sqrt(23) ≈ 5 features par split (randomisation)
            class_weight='balanced',  # compense le déséquilibre Up/Down
            random_state=42,
            n_jobs=-1              # utilise tous les CPU disponibles
        )
        self.rf_model.fit(Xsc, y)

        # Gradient Boosting : plus précis, corrige les erreurs du RF séquentiellement
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,      # 150 arbres boostés
            max_depth=4,           # moins profond que RF (boosting compense)
            learning_rate=0.08,    # pas d'apprentissage conservateur
            subsample=0.8,         # 80% des données par arbre → anti-surapprentissage
            min_samples_leaf=10,
            random_state=42
        )
        self.gb_model.fit(Xsc, y)

        self.model_trained      = True
        self.train_count        += 1
        self.last_train_samples = n

        # Log top-5 features par importance RF (pour analyse sélection de features)
        importances = self.rf_model.feature_importances_
        top_idx     = np.argsort(importances)[::-1][:5]
        top_feat    = [(self.feature_names[i], importances[i]) for i in top_idx]
        self.Debug(f"  Top-5 features : {', '.join([f'{n}={v:.3f}' for n,v in top_feat])}")

    # GÉNÉRATION DES SIGNAUX — Logique Phase 3 (inchangée)

    def GenerateSignals(self):
        """
        Génère les ordres d'achat/vente quotidiennement.

        Pipeline :
        1. Prédiction RF + GB pour chaque action de l'univers
        2. Ensemble : proba_finale = moyenne(proba_RF, proba_GB)
        3. Tri par confiance décroissante (abs(proba - 0.5))
        4. BUY si proba > 0.60 → position sizing proportionnel à la confiance
        5. SELL si proba < 0.40 (signal inverse)

        Position sizing : confidence × 0.5
          - proba=0.60 → confidence=0.10 → target=5% du portfolio
          - proba=0.70 → confidence=0.20 → target=10% du portfolio
          - proba=0.80 → confidence=0.30 → min(15%, max_pos=10%) = 10%
        """
        if self.IsWarmingUp or not self.model_trained:
            return

        # Étape 1 : Prédictions pour toutes les actions prêtes 
        predictions = []
        for symbol, sd in self.symbol_data.items():
            features = self.GetFeaturesForSymbol(symbol)
            if features is None:
                continue
            try:
                X    = np.array(features).reshape(1, -1)
                Xsc  = self.scaler.transform(X)
                rp   = self.rf_model.predict_proba(Xsc)[0]  # [proba_down, proba_up]
                gp   = self.gb_model.predict_proba(Xsc)[0]
                # Ensemble : moyenne des deux modèles → plus stable qu'un seul
                avg  = (rp[1] + gp[1]) / 2
                predictions.append({'symbol': symbol, 'proba_up': avg})
            except:
                continue

        if not predictions:
            return

        # Étape 2 : Tri par confiance décroissante 
        # Les signaux les plus forts sont exécutés en premier
        # Si le budget est épuisé, les signaux faibles sont ignorés
        predictions.sort(key=lambda x: abs(x['proba_up'] - 0.5), reverse=True)

        total_inv   = sum(abs(h.HoldingsValue) for s, h in self.Portfolio.items() if h.Invested)
        pv          = self.Portfolio.TotalPortfolioValue
        cur_inv_pct = total_inv / pv if pv > 0 else 0

        # Étape 3 : Exécution des ordres 
        for pred in predictions:
            symbol   = pred['symbol']
            proba_up = pred['proba_up']
            cur_inv  = self.Portfolio[symbol].Invested
            cur_w    = self.Portfolio[symbol].HoldingsValue / pv if pv > 0 else 0

            # ACHAT : signal haussier avec confiance suffisante 
            if proba_up > self.probability_threshold:
                # Sizing dynamique : plus on est confiant, plus la position est grande
                confidence    = proba_up - 0.5   # [0.10, 0.30] pour proba [0.60, 0.80]
                target_weight = min(self.max_position_weight, confidence * 0.5)
                target_weight = max(0.02, target_weight)  # minimum 2% pour éviter les micro-trades

                # Vérification budget : ne pas dépasser 90% investi total
                if cur_inv_pct + target_weight - abs(cur_w) > self.max_total_invested:
                    continue

                # Exécuter seulement si le changement est significatif (> 2%)
                if abs(target_weight - cur_w) > 0.02:
                    was_invested = cur_inv
                    self.SetHoldings(symbol, target_weight)

                    # Enregistrer l'ouverture pour le Trade Log
                    if not was_invested:
                        price = self.Securities[symbol].Price
                        self.open_trade_probas[symbol] = {
                            'proba':       proba_up,
                            'entry_price': price
                        }

            # VENTE : signal baissier (proba < 0.40) 
            elif proba_up < (1 - self.probability_threshold) and cur_inv:
                self.Liquidate(symbol)
                # OnOrderEvent() se charge du Trade Log automatiquement

    # FEATURE EXTRACTION

    def GetFeaturesForSymbol(self, symbol):
        """
        Retourne le vecteur de 23 features dans l'ordre fixe de feature_names.
        Remplace les valeurs None/NaN/Inf par 0.0 (valeur neutre).
        Retourne None si l'action n'est pas prête.
        """
        if symbol not in self.symbol_data:
            return None
        fd = self.symbol_data[symbol].GetFeatures()
        if fd is None:
            return None

        vector = []
        for name in self.feature_names:
            val = fd.get(name, 0.0)
            if val is None or np.isnan(val) or np.isinf(val):
                val = 0.0
            vector.append(val)
        return vector

    # CONSTRUCTION DU DATASET D'ENTRAÎNEMENT

    def GetTrainingData(self):
        """
        Construit le dataset (X, y) pour l'entraînement des modèles ML.

        Pour chaque action prête :
          - Récupère lookback_days + 60 jours d'historique via History()
          - Pour chaque jour i, calcule les 23 features (même logique que GetFeatures)
          - Label y = 1 si le prix monte dans prediction_horizon jours, 0 sinon
          - Filtre les vecteurs avec NaN/Inf

        Retourne (X, y, None, feature_names) ou (None, None, None, None)
        """
        all_X, all_y = [], []

        for symbol, sd in self.symbol_data.items():
            if not sd.is_ready:
                continue

            # Récupère l'historique (plus long que lookback pour avoir des données suffisantes)
            history = self.History(symbol, self.lookback_days + self.prediction_horizon + 60, Resolution.Daily)
            if history.empty:
                continue

            try:
                df = history.loc[symbol] if symbol in history.index.get_level_values(0) else None
                if df is None or len(df) < self.lookback_days:
                    continue
            except:
                continue

            cl = df['close'].values
            hi = df['high'].values  if 'high'   in df.columns else cl
            lo = df['low'].values   if 'low'    in df.columns else cl
            vo = df['volume'].values if 'volume' in df.columns else np.ones(len(cl))

            # Génération des échantillons (chaque jour = 1 ligne du dataset)
            for i in range(200, len(cl) - self.prediction_horizon):
                try:
                    p, v, c = cl[:i+1], vo[:i+1], cl[i]
                    f = {}

                    # Calcul des 23 features (identique à GetFeatures() mais sur données historiques)
                    f['return_1d']  = (p[-1]-p[-2])/p[-2]
                    f['return_5d']  = (p[-1]-p[-6])/p[-6]
                    f['return_10d'] = (p[-1]-p[-11])/p[-11]
                    f['return_20d'] = (p[-1]-p[-21])/p[-21]

                    r20 = [(p[-20+j]-p[-21+j])/p[-21+j] for j in range(20)]
                    f['volatility_20d'] = float(np.std(r20))

                    tr = [max(hi[j]-lo[j], abs(hi[j]-cl[j-1]) if j>0 else 0,
                              abs(lo[j]-cl[j-1]) if j>0 else 0)
                          for j in range(max(0, i-13), i+1)]
                    f['atr_norm'] = np.mean(tr) / c if c > 0 else 0

                    deltas = [p[-14+j]-p[-15+j] for j in range(14)]
                    ag  = np.mean([max(0, d) for d in deltas])
                    al  = np.mean([max(0, -d) for d in deltas])
                    rsi = 100 - (100 / (1 + ag/al)) if al > 0 else 100
                    f['rsi_norm'] = (rsi - 50) / 50

                    ema12 = np.mean(p[-12:])
                    ema26 = np.mean(p[-26:])
                    f['macd_norm']      = (ema12 - ema26) / c
                    f['macd_hist_norm'] = f['macd_norm'] * 0.7  # approximation histogramme

                    bm = np.mean(p[-20:])
                    bs = np.std(p[-20:])
                    br = 4 * bs
                    f['bb_percent'] = (c - (bm - 2*bs)) / br if br > 0 else 0.5
                    f['bb_width']   = br / bm if bm > 0 else 0

                    s20  = np.mean(p[-20:])
                    s50  = np.mean(p[-50:])
                    s200 = np.mean(p[-200:])
                    f['ma_ratio_20_50']  = s20  / s50  if s50  > 0 else 1
                    f['ma_ratio_50_200'] = s50  / s200 if s200 > 0 else 1
                    f['adx_norm'] = min(1.0, f['volatility_20d'] * 10)  # approximation ADX

                    h14 = max(hi[max(0, i-13):i+1])
                    l14 = min(lo[max(0, i-13):i+1])
                    f['stoch_norm'] = ((c - l14) / (h14 - l14) * 100 - 50) / 50 if h14 > l14 else 0

                    f['momentum_60d']  = (p[-1] - p[-61]) / p[-61]
                    r60 = [(p[-60+j]-p[-61+j])/p[-61+j] for j in range(60)]
                    f['volatility_60d'] = float(np.std(r60))

                    av20 = np.mean(v[-20:])
                    f['volume_ratio'] = v[-1] / av20 if av20 > 0 else 1
                    f['volume_trend'] = np.mean(v[-5:]) / av20 if av20 > 0 else 1

                    f['pe_norm']          = np.log(sd.pe_ratio)/np.log(50) if sd.pe_ratio and 0<sd.pe_ratio<200 else 0.0
                    f['pb_norm']          = np.log(sd.pb_ratio+1)/np.log(20) if sd.pb_ratio and 0<sd.pb_ratio<50 else 0.0
                    f['roe_norm']         = max(-0.5, min(0.5, sd.roe)) if sd.roe is not None else 0.0
                    f['debt_equity_norm'] = min(2.0, sd.debt_equity)/2.0 if sd.debt_equity is not None and sd.debt_equity>=0 else 0.0

                    vec = [f.get(name, 0.0) for name in self.feature_names]
                    if any(np.isnan(x) or np.isinf(x) for x in vec):
                        continue  # ignore les vecteurs corrompus

                    # Label : 1 si prix monte dans prediction_horizon jours
                    future_ret = (cl[i + self.prediction_horizon] - c) / c
                    all_X.append(vec)
                    all_y.append(1 if future_ret > 0 else 0)

                except:
                    continue

        if len(all_X) < 100:
            return None, None, None, None

        return np.array(all_X), np.array(all_y), None, self.feature_names

    # RAPPORT FINAL

    def OnEndOfAlgorithm(self):
        """
        Appelé automatiquement par LEAN à la fin du backtest.
        Affiche les métriques clés et le tableau win rate par niveau de confiance.
        """
        self.Log("\n" + "=" * 60)
        self.Log("RÉSULTATS FINAUX — ML ALPHA v8 FINALE")
        self.Log("=" * 60)

        pv     = self.Portfolio.TotalPortfolioValue
        n_yrs  = (self.EndDate - self.StartDate).days / 365.25
        profit = (pv / 100000 - 1) * 100
        ann    = ((pv / 100000) ** (1 / n_yrs) - 1) * 100

        self.Log(f"  Période        : {self.StartDate.date()} → {self.EndDate.date()}")
        self.Log(f"  Valeur finale  : ${pv:,.0f}")
        self.Log(f"  Rendement tot. : {profit:.1f}%")
        self.Log(f"  CAGR           : {ann:.1f}% / an")
        self.Log(f"  Retrains       : {self.train_count} (mensuel)")

        if self.wf_accs:
            self.Log(f"  WF-OOS acc     : {np.mean(self.wf_accs):.3f} ± {np.std(self.wf_accs):.3f}")
            self.Log(f"  WF-OOS prec    : {np.mean(self.wf_precs):.3f}")
            self.Log(f"  WF-OOS f1      : {np.mean(self.wf_f1s):.3f}")
            self.Log(f"  Nb folds WF    : {len(self.wf_accs)}")

        # Tableau win rate par niveau de confiance
        self.Log(f"\n--- WIN RATE PAR NIVEAU DE CONFIANCE ML ---")
        self.Log(f"  {'Bucket':<14} {'Trades':>7} {'Win Rate':>9} {'PnL moy':>9}")
        self.Log(f"  {'-'*43}")
        total_logged = 0
        for bk, d in self.proba_buckets.items():
            n = d['total']
            if n == 0: continue
            total_logged += n
            wr  = d['wins'] / n * 100
            pnl = d['pnl']  / n * 100
            self.Log(f"  {bk:<14} {n:>7} {wr:>8.1f}% {pnl:>8.2f}%")
        self.Log(f"  {'-'*43}")
        self.Log(f"  Total loggé : {total_logged} trades")

        # Positions finales
        positions = sorted(
            [(s.Value, h.HoldingsValue / pv * 100) for s, h in self.Portfolio.items() if h.Invested],
            key=lambda x: x[1], reverse=True
        )
        self.Log(f"\n  Positions finales ({len(positions)}) :")
        for sym, w in positions[:10]:
            self.Log(f"    {sym}: {w:.1f}%")

        self.Log("=" * 60)
