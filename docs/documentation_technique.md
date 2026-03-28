# Documentation Technique — ML Alpha Strategy

## 1. Plateforme : QuantConnect / LEAN

QuantConnect est une plateforme de backtesting institutionnel utilisant le moteur open-source LEAN. Les données sont fournies par Morningstar (point-in-time) et Equity US (prix OHLCV quotidiens). Le backtesting simule des conditions réelles incluant les frais de courtage Interactive Brokers.

---

## 2. Universe Selection

### CoarseFilter
Filtre sur ~10 000 actions US quotidiennement :
- `HasFundamentalData` : données Morningstar disponibles (corrige partiellement le survivorship bias)
- `Price > $10` : élimine les penny stocks
- `DollarVolume > $5M/jour` : liquidité suffisante pour exécution réelle
- Sélection des **150 plus liquides** pour pré-filtrage

### FineFilter
Filtre sur les 150 pré-sélectionnés :
- `MarketCap > $2Mds` : mid/large cap uniquement
- Sélection des **50 plus grandes capitalisations**
- Mise à jour des fondamentaux (PE, PB, ROE, D/E) via Morningstar

---

## 3. Feature Engineering — 23 features

### Catégorie 1 : Returns multi-timeframe (4 features)
```
return_1d   = (close[-1] - close[-2]) / close[-2]
return_5d   = (close[-1] - close[-6]) / close[-6]
return_10d  = (close[-1] - close[-11]) / close[-11]
return_20d  = (close[-1] - close[-21]) / close[-21]
```

### Catégorie 2 : Volatilité (2 features)
```
volatility_20d = std(daily_returns[-20:])
atr_norm       = ATR(14) / close_price
```

### Catégorie 3 : Oscillateurs techniques (5 features)
```
rsi_norm       = (RSI(14) - 50) / 50              # [-1, 1]
macd_norm      = MACD(12,26,9) / close_price
macd_hist_norm = (MACD - Signal) / close_price
bb_percent     = (close - BB_lower) / (BB_upper - BB_lower)
bb_width       = (BB_upper - BB_lower) / BB_middle
```

### Catégorie 4 : Trend / Momentum (4 features)
```
ma_ratio_20_50  = SMA(20) / SMA(50)     # >1 = tendance haussière court terme
adx_norm        = ADX(14) / 100
stoch_norm      = (StochK - 50) / 50
momentum_60d    = (close[-1] - close[-61]) / close[-61]
```

### Catégorie 5 : Long terme (2 features)
```
volatility_60d  = std(daily_returns[-60:])
ma_ratio_50_200 = SMA(50) / SMA(200)    # golden cross indicator
```

### Catégorie 6 : Volume (2 features)
```
volume_ratio = volume[-1] / mean(volume[-20:])
volume_trend = mean(volume[-5:]) / mean(volume[-20:])
```

### Catégorie 7 : Fondamentaux (4 features)
```
pe_norm          = log(PE) / log(50)        si 0 < PE < 200
pb_norm          = log(PB+1) / log(20)      si 0 < PB < 50
roe_norm         = clamp(ROE, -0.5, 0.5)
debt_equity_norm = min(D/E, 2.0) / 2.0
```

---

## 4. Modèle ML Ensemble

### Random Forest
```python
RandomForestClassifier(
    n_estimators=200,       # 200 arbres de décision
    max_depth=5,            # profondeur limitée → anti-overfitting
    min_samples_leaf=10,    # régularisation
    max_features='sqrt',    # sqrt(23) ≈ 5 features par split
    class_weight='balanced' # compense déséquilibre Up/Down
)
```

### Gradient Boosting
```python
GradientBoostingClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.08,     # pas conservateur
    subsample=0.8           # stochastic boosting → anti-overfitting
)
```

### Ensemble
```
proba_finale = (proba_RF + proba_GB) / 2
Signal BUY   si proba_finale > 0.60
Signal SELL  si proba_finale < 0.40
```

---

## 5. Walk-Forward Validation

Chaque mois au moment du retrain :
```
Données disponibles : N échantillons
  ├── Train : [N - 378 - 21 : N - 21]  → 378 jours
  └── Test  : [N - 21 : N]             → 21 jours (jamais vus)

Métriques OOS calculées : accuracy, precision, F1-score
```
Accuracy OOS observée : 0.52 — 0.58 (selon les périodes)

---

## 6. Position Sizing

```
confidence    = proba_up - 0.5                        # [0.10, 0.30]
target_weight = min(max_position, confidence × 0.5)   # [2%, 10%]

Exemples :
  proba = 0.60 → confidence = 0.10 → weight = 5%
  proba = 0.70 → confidence = 0.20 → weight = 10%
  proba = 0.80 → confidence = 0.30 → min(15%, 10%) = 10%
```

---

## 7. Risk Management

1. **Stop-loss natif LEAN** : `MaximumDrawdownPercentPerSecurity(0.05)` → liquidation automatique à -5%
2. **Budget total** : max 90% investi simultanément
3. **Diversification** : max 10% par action
4. **Ordre d'exécution** : signaux triés par confiance décroissante → les meilleurs passent en premier

---

## 8. Découvertes techniques importantes

### Warmup QC
- `SetWarmup(int)` compte des jours de marché depuis `SetStartDate`, pas des jours calendaires
- L'univers dynamique n'est pas inclus dans le warmup → `is_ready` détermine le vrai délai
- **Formule empirique** : `SetStartDate + 18 mois ≈ premier trade` (avec lookback 378j)

### Trade Log
- `OnOrderEvent()` capture TOUTES les fermetures (signal ML + stop LEAN + liquidation)
- Un trade log manuel `_CloseTradeLog()` manquait les stops → faux win rate de 85-97%
- Win rate réel avec `OnOrderEvent` : 53-62% selon les périodes

### Lookback optimal
- LB252 (1 an) : Sharpe moyen 0.482, 1/6 périodes > SPY
- LB378 (1.5 an) : Sharpe moyen 0.553, 5/6 périodes > SPY
- Interprétation : plus de données d'entraînement → modèle plus robuste

### Path dependency
- 3 mois de décalage de démarrage peuvent impacter le résultat de +47% sur 5 ans
- Confirme l'importance des tests de robustesse multi-périodes