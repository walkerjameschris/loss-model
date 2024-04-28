# Loss Model

## Introduction

A repo containing code to estimate a mortgage loss model using open source loss
data from the Freddie Mae loan-level mortgage performance data repository. The 
code base compares machine learning methods ranging from OLS and GAMs to XGBoost
and Neural Nets.

## Contents

- `analysis/`
  - `01_wrangle.R`: Unpacks Freddie Mac data and constructs loan level data
  - `02_preprocess.R`: Preprocesses complete data for modeling applications
  - `03_estimate.R`: Estimates many variants of a loss model
- `data/`
  - `base_rec.Rds`: The baseline model recipe
  - `complete_data.Rds`: A complete data set before preprocessing or splits
  - `performance.csv`: A performance summary of all models
  - `specs.Rds`: Serialized specs for each model
  - `stability.csv`: A coefficient stability exercise
  - `train_test.Rds`: Train, test, and validate serialized as a list
  - `tune_nn.csv`: Results of an neural network tuning exercise
  - `tune_xgb.csv`: Results of an XGB tuning exercise
- `loss-model.Rproj`: Sets the working directory
- `report/`
  - `report.qmd`: Final report as PDF document
  - `report.qmd`: Final report as HTML document
- `src/`
  - `dials.yaml`: Project level settings
  - `functions.R`: Functions used across the codebase
