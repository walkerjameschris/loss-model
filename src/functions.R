
#### Data Preparation ####

dials <- yaml::read_yaml(here::here("src/dials.yaml"))

get_zip_info <- function(files, dials, keep_cols, time_file) {
  # Grabs file components
  
  f <- purrr::discard
  
  if (time_file) {
    f <- purrr::keep
  }
  
  # Determine if we want time file or orig file
  file <- f(files, \(x) stringr::str_detect(x, "time"))
  
  # Create file nice names
  pattern <- "^\\./|_[0-9]{4}Q[0-9]\\.txt"
  refer <- stringr::str_remove_all(file, pattern)
  dials_subset <- unlist(dials[[refer]])
  col_names <- names(dials_subset)
  
  # Return components
  tibble::lst(
    path = here::here(file),
    index = which(col_names %in% cols),
    names = col_names[index],
    types = unname(dials_subset)
  )
}

factorize <- function(x, levels, fun = tolower) {
  # Allows for easy leveling factors
  
  x <- fun(x)
  x <- dplyr::if_else(x %in% levels, x, "ref")
  factor(x, c("ref", levels))
}

#### Modeling and Validation ####

t_test <- function(x, level = 0.95) {
  # Simple t-test utility
  
  stats::t.test(
    x = x,
    conf.level = level
  ) |>
    purrr::pluck("conf.int") |>
    as.list() |>
    purrr::set_names(
      c("lo", "hi")
    ) |>
    tibble::as_tibble()
}

gini_vec <- function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
  truth <- factor(dplyr::percent_rank(truth) < 0.95)
  2 * yardstick::roc_auc_vec(truth, estimate) - 1
}

gini <-
  yardstick::new_numeric_metric(
    fn = \(data, ...) UseMethod("gini"),
    direction = "maximize"
  )

gini.data.frame <- function(data,
                            truth,
                            estimate,
                            na_rm = TRUE,
                            case_weights = NULL,
                            ...) {
  
  yardstick::numeric_metric_summarizer(
    name = "gini",
    fn = gini_vec,
    data = data,
    truth = {{ truth }},
    estimate = {{ estimate }},
    na_rm = na_rm,
    case_weights = {{ case_weights }}
  )
}

tmr_vec <- function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
  truth <- dplyr::percent_rank(truth) > 0.95
  estimate <- dplyr::percent_rank(estimate) > 0.95
  sum(truth * estimate) / sum(truth)
}

tmr <-
  yardstick::new_numeric_metric(
    fn = \(data, ...) UseMethod("tmr"),
    direction = "maximize"
  )

tmr.data.frame <- function(data,
                           truth,
                           estimate,
                           na_rm = TRUE,
                           case_weights = NULL,
                           ...) {
  
  yardstick::numeric_metric_summarizer(
    name = "tmr",
    fn = tmr_vec,
    data = data,
    truth = {{ truth }},
    estimate = {{ estimate }},
    na_rm = na_rm,
    case_weights = {{ case_weights }}
  )
}

metric_bundle <- yardstick::metric_set(gini, tmr, yardstick::rmse)

get_performance <- function(data,
                            pred = .pred,
                            truth = gross_loss,
                            population = population) {
  # A wrapper using metric sets and augment
  
  data |>
    dplyr::group_by(
      {{ population }}
    ) |>
    metric_bundle(
      truth = {{ truth }},
      estimate = {{ pred }}
    ) |>
    dplyr::select(- .estimator) |>
    tidyr::pivot_wider(
      names_from = .metric,
      values_from = .estimate
    )
}

make_predictions <- function(data, model) {
  # Make predictions
  
  dplyr::bind_rows(
    data,
    .id = "population"
  ) |>
    parsnip::augment(
      x = model,
      new_data = _
    )
}

evaluate <- function(model,
                     data,
                     truth = gross_loss,
                     population = population) {
  # Full evaluation of model
  
  make_predictions(
    data = data,
    model = model
  ) |>
    get_performance(
      truth = {{ truth }},
      population = {{ population }}
    )
}

gam_formula <- function(recipe, terms = c("fico", "dti", "cltv")) {
  
  recipes::prep(recipe) |>
    purrr::pluck("term_info") |>
    dplyr::filter(
      role %in% c("outcome", "predictor")
    ) |>
    dplyr::mutate(
      variable = dplyr::if_else(
        variable %in% terms,
        glue::glue("s({variable})"),
        glue::as_glue(variable)
      )
    ) |>
    split(~ role) |>
    purrr::map("variable") |>
    purrr::map(
      \(x) glue::glue_collapse(x, " + ")
    ) |>
    glue::glue_data("{outcome} ~ {predictor}") |>
    as.formula()
}

#### Presentation ####

theme_plot <- function() {
  
  showtext::showtext_auto()
  
  ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "top",
      text = ggplot2::element_text(size = 60),
      plot.title = ggplot2::element_text(face = "bold"),
      legend.title = ggplot2::element_blank(),
      panel.background = ggplot2::element_blank()
    )
}

make_nicenames <- function(x) {
  # A simple name remap utility
  
  dplyr::case_match(
    x,
    
    # Predictors
    "(Intercept)" ~ "Intercept",
    "cltv" ~ "CLTV",
    "dti" ~ "DTI",
    "fico" ~ "FICO",
    "fthb" ~ "FTHB",
    "occupancy_i" ~ "Occupy: Investor",
    "occupancy_s" ~ "Occupy: Second",
    "one_borrower" ~ "One Borrower",
    "one_unit" ~ "Single Family",
    "property_co" ~ "Property: Condo",
    "property_pu" ~ "Property: Planned",
    "purpose_c" ~ "Purpose: Cash Out",
    "purpose_n" ~ "Purpose: Refi",
    "region_northeast" ~ "Region: NE",
    "region_south" ~ "Region: South",
    "region_west" ~ "Region: West",
    "term_frm180" ~ "Term: 15yr",
    "term_frm240" ~ "Term: 20yr",
    "fico_lt_720" ~ "FICO < 720",
    "fico_gt_720" ~ "FICO > 720",
    "cltv_lt_80" ~ "CLTV < 80",
    "cltv_gt_80" ~ "CLTV > 80",
    "dti_lt_37" ~ "DTI < 37",
    "dti_gt_37" ~ "DTI > 37",
    
    # Model Types
    "gam" ~ "Generalized Additive Model",
    "ols" ~ "Linear Regression",
    "xgb" ~ "XGBoost",
    "nn" ~ "Neural Network",
    "torch" ~ "Torch",
    
    # Performance columns
    "gini" ~ "Gini",
    "tmr" ~ "TMR 5%",
    "rmse" ~ "RMSE",
    "population" ~ "Population",
    "model" ~ "Model",
    
    # Population Labels
    "train" ~ "Train",
    "test" ~ "Test",
    "validate" ~ "Validate",
    
    .default = x
  )
}
