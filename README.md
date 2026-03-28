# ML Alpha Strategy — Projet H.1

**ECE Paris — 2026**  
**Équipe :** Antoine Balssa · Antoine Bourdelois · Augustin Couet

---

## Contexte

Développement d'une stratégie de trading algorithmique basée sur le Machine Learning, backtestée sur la plateforme QuantConnect (moteur LEAN). L'objectif est de générer un alpha positif par rapport au benchmark S&P 500 (SPY) sur des périodes glissantes de 5 ans.

---

## Architecture

```
ML Alpha Strategy
├── Universe Selection    → Top 50 actions US par market cap (dynamique, mensuel)
├── Feature Engineering   → 23 features (techniques + fondamentaux)
├── ML Ensemble           → Random Forest (200 arbres) + Gradient Boosting (150 arbres)
├── Walk-Forward OOS      → Validation honnête sur 21j non vus chaque mois
├── Position Sizing       → Proportionnel à la confiance ML (confidence × 0.5)
└── Risk Management       → Stop-loss -5% natif LEAN + budget max 90% investi
```

---

## Installation et usage

### Prérequis
- Compte QuantConnect gratuit : [quantconnect.com](https://www.quantconnect.com)
- Aucune installation locale nécessaire (exécution cloud)

### Lancer un backtest
1. Créer un nouveau projet sur QuantConnect
2. Copier le contenu de `src/main_FINAL.py` dans l'éditeur
3. Modifier les 2 lignes de dates dans `Initialize()` :
```python
self.SetStartDate(2010, 6, 1)   # ← 18 mois avant le premier trade souhaité
self.SetEndDate(2016, 12, 31)   # ← premier trade + 5 ans
```
4. Cliquer sur **Build and Backtest**

### Dates validées pour la démonstration

| Premier trade | SetStartDate | SetEndDate |
|---|---|---|
| ~jan 2010 | (2008, 6, 1) | (2014, 12, 31) |
| ~jan 2012 | (2010, 6, 1) | (2016, 12, 31) |
| ~jan 2014 | (2012, 5, 1) | (2018, 12, 31) |
| ~jan 2016 | (2014, 6, 1) | (2020, 12, 31) |
| ~jan 2018 | (2016, 6, 1) | (2022, 12, 31) |
| ~jan 2020 | (2018, 4, 1) | (2024, 12, 31) |

---

## Résultats (lookback 378j — version finale)

| Période | Sharpe | Drawdown | CAGR | Net Profit | vs SPY |
|---|---|---|---|---|---|
| 2010-2014 | 0.660 | -16.0% | 14.2% | +94.0% | -11.1% |
| 2012-2016 | 0.757 | -16.0% | 16.2% | +111.9% | +13.7% |
| 2014-2018 | 0.467 | -17.1% | 11.3% | +70.7% | +20.4% |
| 2016-2020 | 0.587 | -26.7% | 18.3% | +131.3% | +28.2% |
| 2018-2022 | 0.339 | -27.2% | 11.6% | +73.2% | +16.3% |
| 2020-2024 | 0.511 | -22.8% | 19.2% | +140.8% | +43.8% |

**Sharpe moyen : 0.553 | 5/6 périodes battent le SPY**

---

## Paramètres clés

| Paramètre | Valeur | Justification |
|---|---|---|
| `lookback_days` | 378 | Optimal vs 252j — testé sur 6 périodes |
| `num_stocks` | 50 | Top 50 US par market cap |
| `probability_threshold` | 0.60 | Seuil confiance ML minimum |
| `max_position_weight` | 10% | Diversification forcée |
| `max_total_invested` | 90% | Réserve cash de sécurité |
| `prediction_horizon` | 5 jours | Horizon de prédiction ML |

---

## Structure du projet

```
groupe-XX-ml-alpha/
├── README.md                  ← ce fichier
├── src/
│   └── main_FINAL.py          ← code source principal (version finale)
├── docs/
│   └── documentation_technique.md
└── slides/
    └── [lien ou PDF à ajouter]
```

---

## Limites connues

- **Biais de survivorship** : partiellement corrigé par les données point-in-time Morningstar de QC. Biais résiduel estimé à 2-5%.
- **Path dependency** : 3 mois de décalage au démarrage peuvent impacter significativement les résultats (effet 2013 +32%).
- **Sensibilité au régime** : 2018-2022 (inflation + hausse des taux) est la période la plus difficile (Sharpe 0.339).
- **Walk-Forward accuracy** : 0.52-0.58 — faible mais honnête et suffisant avec un P/L ratio > 1.1.