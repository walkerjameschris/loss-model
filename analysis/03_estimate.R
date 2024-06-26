#### Setup ####

# 03_estimate.R
# Chris Walker

# Estimates many variants of a loss model

library(withr)
library(purrr)
library(readr)
library(dplyr)
library(agua)
library(xgboost)
library(tidymodels)

source(here::here("src/functions.R"))

train_test <-
  readr::read_rds(
    here::here("data/train_test.Rds")
  )

base_rec <-
  readr::read_rds(
    here::here("data/base_rec.Rds")
  )

#### Assess Stability ####

lm_spec <- parsnip::linear_reg()
stable_wflow <- workflows::workflow(base_rec, lm_spec)

if (dials$run_stable) {
  
  rsample::bootstraps(
    data = train_test$train,
    times = dials$n_stability
  ) |>
    purrr::pmap(function(splits, id) {
      
      stable_wflow |>
        parsnip::fit(
          rsample::analysis(splits)
        ) |>
        broom::tidy()
      
    }, .progress = TRUE) |>
    withr::with_seed(
      seed = 12345
    ) |>
    dplyr::bind_rows() |>
    readr::write_csv(
      here::here("data/stability.csv")
    )
  
}

#### OLS with Earth Splines ####

model_rec <-
  recipes::step_rm(
    recipe = base_rec,
    property_co,
    term_frm180,
    region_northeast
  )

if (dials$run_mars) {
  
  parsnip::mars(
    mode = "regression",
    engine = "earth"
  ) |>
    workflows::workflow(
      spec = _,
      preprocessor = model_rec
    ) |>
    parsnip::fit(
      train_test$train
    ) |>
    parsnip::extract_fit_engine() |>
    purrr::pluck("coefficients") |>
    print()
  
}

ols_rec <-
  recipes::step_mutate(
    recipe = model_rec,
    fico_lt_720 = pmin(7.2, fico) - 7.2,
    fico_gt_720 = pmax(7.2, fico) - 7.2,
    cltv_lt_80 = pmin(8, cltv) - 8,
    cltv_gt_80 = pmax(8, cltv) - 8
  ) |>
  recipes::step_rm(
    fico,
    cltv
  )

ols_mod <-
  workflows::workflow(
    ols_rec,
    parsnip::linear_reg()
  ) |>
  parsnip::fit(train_test$train)

ols <- evaluate(ols_mod, train_test)

#### XGBoost ####

bootstrap_data <-
  rsample::bootstraps(
    data = train_test$train,
    times = 3
  ) |>
  withr::with_seed(
    seed = 12345
  )

xgb_grid <-
  tidyr::expand_grid(
    tree_depth = dials$tree_depth,
    learn_rate = dials$learn_rate,
    trees = dials$trees
  )

xgb_wflow <-
  parsnip::boost_tree(
    mode = "regression",
    engine = "xgboost",
    trees = tune::tune(),
    tree_depth = tune::tune(),
    learn_rate = tune::tune()
  ) |>
  workflows::workflow(
    preprocessor = model_rec,
    spec = _
  )

if (dials$tune_xgb) {
  
  tune::tune_grid(
    object = xgb_wflow,
    resamples = bootstrap_data,
    metrics = metric_bundle,
    grid = xgb_grid,
    control = tune::control_grid(verbose = TRUE)
  ) |>
    tune::collect_metrics() |>
    dplyr::select(
      dplyr::all_of(names(xgb_grid)),
      metric = .metric,
      mean
    ) |>
    readr::write_csv(
      here::here("data/tune_xgb.csv")
    )
  
}

xgb_mod <-
  list(
    tree_depth = dials$xgb_depth,
    learn_rate = dials$xgb_rate,
    trees = dials$xgb_rounds
  ) |>
  tune::finalize_workflow(
    x = xgb_wflow,
    parameters = _
  ) |>
  parsnip::fit(train_test$train)

xgb <- evaluate(xgb_mod, train_test)

xgb_imp <-
  parsnip::extract_fit_engine(xgb_mod) |>
  xgboost::xgb.importance(model = _) |>
  tibble::as_tibble() |>
  dplyr::rename_all(tolower)

#### Neural Network ####

mlp_rec <-
  recipes::step_normalize(
    recipe = model_rec,
    fico, dti, cltv
  )

agua::h2o_start()

mlp_grid <-
  tidyr::expand_grid(
    hidden_units = dials$hidden_units,
    epochs = dials$epochs
  )

mlp_wflow <-
  parsnip::mlp(
    mode = "regression",
    hidden_units = tune::tune(),
    epochs = tune::tune(),
    activation = "tanh"
  ) |>
  parsnip::set_engine(
    engine = "h2o",
    reproducible = TRUE, 
    seed = 2468
  ) |>
  workflows::workflow(
    preprocessor = mlp_rec,
    spec = _
  )

if (dials$tune_mlp) {
    
  tune::tune_grid(
    object = mlp_wflow,
    resamples = bootstrap_data,
    metrics = metric_bundle,
    grid = mlp_grid,
    control = tune::control_grid(verbose = TRUE)
  ) |>
    tune::collect_metrics() |>
    dplyr::select(
      dplyr::all_of(names(mlp_grid)),
      metric = .metric,
      mean
    ) |>
    readr::write_csv(
      here::here("data/tune_mlp.csv")
    )

}

mlp_mod <-
  list(
    hidden_units = dials$mlp_nodes,
    epochs = dials$mlp_epochs
  ) |>
  tune::finalize_workflow(
    x = mlp_wflow,
    parameters = _
  ) |>
  parsnip::fit(train_test$train)

mlp <- evaluate(mlp_mod, train_test)

#### Deep Learning with Python and Torch ####

temp_files <-
  train_test |>
  purrr::imap_chr(function(data, i) {
    
    temp_path <- here::here(i)
    
    recipes::bake(
      object = recipes::prep(mlp_rec),
      new_data = data
    ) |>
      readr::write_csv(temp_path)
    
    temp_path
  })

reticulate::py_run_file(
  here::here("src/deep_learning.py")
)

torch <-
  purrr::map(
    temp_files,
    readr::read_csv
  ) |>
  dplyr::bind_rows() |>
  get_performance()

fs::file_delete(temp_files)

#### Merge Performance ####

tibble::lst(
  ols,
  xgb,
  mlp,
  torch
) |>
  dplyr::bind_rows(
    .id = "model"
  ) |>
  dplyr::relocate(
    population,
    model,
    gini,
    tmr
  ) |>
  dplyr::arrange(
    population != "train",
    desc(gini)
  ) |>
  print(
    n = Inf
  ) |>
  readr::write_csv(
    here::here("data/performance.csv")
  )

tibble::lst(
  ols = broom::tidy(ols_mod),
  ols_resid = parsnip::augment(ols_mod, head(train_test$test, 1000)),
  xgb = xgb_imp
) |>
  readr::write_rds(
    here::here("data/specs.Rds")
  )
