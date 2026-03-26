# Win Bundesliga - Fantasy League Prediction Model

A machine learning project that predicts Bundesliga match outcomes to gain an edge in your fantasy league competition with friends.

## Overview

This project uses XGBoost classification to predict whether the home team will win, draw, or lose each Bundesliga match. The model analyzes historical match data including team form, recent performance metrics, and bookmaker odds to make informed predictions.

**Status:** Work in progress. Model currently shows ~63% average accuracy across time-series validation folds, but exhibits overfitting. Regularization improvements are ongoing.

## Features & Methodology

### Input Features

The model uses 8 key features:
- **Time Slot** - Encoded match time (maps each unique kickoff time to a numerical value)
- **Home Team & Away Team** - Team identifiers (encoded 0-17 for 18 Bundesliga teams)
- **Goal Differential (Last 5 Games)** - Home team average goals minus away team average goals in their last 5 matches
- **Team Form Differential (Last 3 Games)** - Points earned by home team minus away team points in last 3 matches
- **Bookmaker Odds** - Win, draw, and loss odds from betting aggregators

### Feature Engineering

The model creates rolling statistics from historical data:
- 5-game rolling averages for goals scored and shots taken
- 3-game rolling form points (wins=3, draws=1, losses=0)
- Historical data decay weighting (older matches weighted less than recent ones)

### Target Variable

Three-class classification:
- 0 = Away win (A)
- 1 = Draw (D)
- 2 = Home win (H)

## Model Configuration

**Algorithm:** XGBoost Classifier
- Estimators: 500
- Max Depth: 2 (reduces overfitting)
- Learning Rate: 0.05
- Objective: Multi-class softprob
- Evaluation Metric: Multi-class log loss

**Validation:** Time-Series Cross-Validation (20 splits)
- Ensures temporal integrity - model only sees historical data
- Prevents information leakage from future matches

**Sample Weighting:** Exponential decay by age
- Formula: `exp(-days_since_match / 100)`
- Recent matches have higher influence on model training

## Performance Results

Time-series cross-validation results (20 folds):

| Metric | Value |
|--------|-------|
| Average Accuracy | 63.34% |
| Average Log Loss | 0.9362 |
| Accuracy Range | 33.33% - 77.78% |
| Log Loss Range | 0.5727 - 1.2873 |

**Key Observation:** High variance across folds indicates the model may be overfitting to specific time periods or match conditions.

## Data Source

Match data downloaded from **football-data.co.uk**:
- Bundesliga (D1) historical match data
- Includes match results and betting odds from multiple bookmakers
- Simple CSV download (no web scraping required)

## Known Issues & Improvements

### Current Issues
- Overfitting on training data
- High variance in cross-validation performance across folds
- Model struggles with certain time periods or team matchups

### Planned Improvements
1. Implement additional regularization 
2. Feature selection to reduce dimensionality
4. Incorporate additional features (team injuries, recent transfers, head-to-head history)
5. Hyperparameter tuning 
6. Early stopping with more rigorous validation
---

**Data Source:** football-data.co.uk - Bundesliga matches and odds
