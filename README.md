# EPL Match Outcome Prediction

A machine learning project that predicts English Premier League match outcomes (win, draw, loss) using historical performance data, with uncertainty analysis via prediction entropy.

## Dataset

Kaggle — EPL Match Data 2000–2025 (~9,380 matches across 25 seasons)

## Features

All features are computed strictly from data prior to each match to prevent leakage.

| Feature | Description |
|---|---|
| `home_ppg_season` | Home team points per game, season to date |
| `home_gd_per_game_season` | Home team goal difference per game, season to date |
| `home_ppg_home_only` | Home team points per game in home matches only |
| `home_ppg_last5` | Home team points per game, last 5 matches |
| `home_goals_scored_last5` | Home team avg goals scored, last 5 matches |
| `home_goals_conceded_last5` | Home team avg goals conceded, last 5 matches |
| `away_ppg_season` | Away team points per game, season to date |
| `away_gd_per_game_season` | Away team goal difference per game, season to date |
| `away_ppg_away_only` | Away team points per game in away matches only |
| `away_ppg_last5` | Away team points per game, last 5 matches |
| `away_goals_scored_last5` | Away team avg goals scored, last 5 matches |
| `away_goals_conceded_last5` | Away team avg goals conceded, last 5 matches |
| `h2h_home_winrate` | Home team historical win rate vs. this opponent |
| `h2h_home_avg_gd` | Home team historical avg goal difference vs. this opponent |

**Target:** `result` — W / D / L from the home team's perspective

## Models

- Logistic Regression
- XGBoost
- Neural Network

## Project Structure

```
epl-match-prediction/
├── data/
│   ├── raw/                       # Original Kaggle dataset
│   └── epl_features.csv           # Engineered feature dataset
├── notebooks/
│   ├── 01_feature_engineering.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
└── README.md
```

## Requirements

```
pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Authors

Ben Patte & Kerem Atas
