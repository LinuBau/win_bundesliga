# Win Bundesliga - Fantasy League Prediction Model

A machine learning project that predicts Bundesliga match outcomes to gain an edge in your fantasy league competition with friends.

## Overview

This project uses XGBoost classification to predict whether the home team will win, draw, or lose each Bundesliga match. The model analyzes historical match data including team form, recent performance metrics, and bookmaker odds to make informed predictions.

**Status:** Work in progress. Model currently shows ~63% average accuracy across time-series validation folds, but exhibits overfitting. Regularization improvements are ongoing.

## Features & Methodology

### Input Features

The model uses 8 key features:
- **Time Slot** - Encoded match time (maps each unique kickoff time to a numerical value)
- **Home Team & Away Team** - Team identifiers (encoded 0-17 for 18 Bundesliga teams) //not in use Data Labels to big makes the Modelle unstable
- **Goal Differential (Last 5 Games)** - Home team average goals minus away team average goals in their last 5 matches
- **Team Form Differential (Last 5 Games)** - Points earned by home team minus away team points in last 3 matches
- **Bookmaker Odds** - Win, draw, and loss odds from betting aggregators

### Feature Engineering

The model creates rolling statistics from historical data:
- 5-game rolling averages for goals scored and shots taken
- 5-game rolling form points (wins=3, draws=1, losses=0)
- Historical data decay weighting (older matches weighted less than recent ones)

### Target Variable

Three-class classification:
- 0 = Away win (A)
- 1 = Draw (D)
- 2 = Home win (H)

## Model Configuration

**Algorithm:** XGBoost Classifier
- Estimators: 1000
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
| Average Accuracy | 58.82% |
| Average Log Loss | 0.0.9996 |
| Accuracy Range | % - 775490.78% |
| Log Loss Range | 0.5727 - 1.2873 |

**Key Observation:** High variance across folds indicates the model may be overfitting to specific time periods or match conditions.

## Data Source

Match data downloaded from **football-data.co.uk**:
- Bundesliga (D1) historical match data
- Includes match results and betting odds from multiple bookmakers
- Simple CSV download (no web scraping required)

## Known Issues & Improvements

### Current Issues

This issue cannot be fixed without increasing the dataset or adding meaningful features, which isn't possible with the dataset I'm currently using. I need features that aren't just bookmaker odds to prevent the model from simply learning to follow the bookmakers, as that isn't my goal. Using data that contains deeper features like expected goals (xG) increases the likelihood of finding a genuine edge. However, while adding more data decreased the overall overfitting and added mathematical stability, the model's true accuracy actually dropped below 50%. This feels completely counterproductive, as it performs worse than a coin flip. Because of this, I plan to try this again from scratch using a better, data-mined dataset.

### Planned Improvements
Ussing new Dataset that is data-mined.
---

**Data Source:** football-data.co.uk - Bundesliga matches and odds
