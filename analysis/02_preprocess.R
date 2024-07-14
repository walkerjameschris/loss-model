#### Setup ####

# 02_preprocess.R
# Chris Walker

# Preprocesses complete data for use in modeling applications

library(dplyr)
library(readr)
library(tidymodels)

source(here::here("src/functions.R"))

#### Split Data and Create Recipe ####

complete_data <-
  readr::read_rds(
    here::here("data/complete_data.Rds")
  )

init_split <-
  complete_data |>
  dplyr::filter(
    dti != 999,
    cltv != 999,
    fico != 9999,
    one_borrower != 999
  ) |>
  rsample::initial_split(
    prop = 0.75
  ) |>
  withr::with_seed(
    seed = 321
  )

train <- rsample::training(init_split)
test <- rsample::testing(init_split)

tibble::lst(
  train,
  test
) |>
  readr::write_rds(
    file = here::here("data/train_test.Rds"),
    compress = "xz"
  )

recipes::recipe(
  gross_loss ~ .,
  data = train
) |>
  recipes::step_mutate(
    
    # Switch state to region
    region = stringr::str_replace_all(
      string = state,
      pattern = purrr::set_names(
        as.character(state.region),
        as.character(state.abb)
      )
    ),
    
    # Coerce term values
    term = dplyr::case_when(
      term < 180 ~ "frm180",
      term < 240 ~ "frm240",
      term < 360 ~ "frm360",
      .default = "frm480"
    ),
    
    # Make continuous factors more sensible
    cltv = cltv / 10,
    fico = fico / 100,
    dti = dti / 10,
    
    # Single level factors
    fthb = 1 * (fthb %in% "Y"),
    one_unit = 1 * (one_unit %in% "1"),
    one_borrower = 1 * (one_borrower %in% "1"),
    
    # Multiple level categories
    occupancy = factorize(occupancy, c("i", "s")),
    property = factorize(property, c("co", "pu")),
    purpose = factorize(purpose, c("c", "n")),
    term = factorize(term, c("frm180", "frm240")),
    region = factorize(region, c("west", "south", "northeast"))
    
  ) |>
  recipes::step_rm(
    upb,
    state
  ) |>
  recipes::step_dummy(
    recipes::all_nominal_predictors()
  ) |>
  readr::write_rds(
    file = here::here("data/base_rec.Rds"),
    compress = "xz"
  )
