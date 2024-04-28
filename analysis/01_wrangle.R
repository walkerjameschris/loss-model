#### Setup ####

# 01_wrangle.R
# Chris Walker

# Unpacks Freddie Mac performance sample and constructs loan level data

library(dplyr)
library(readr)

source(here::here("src/functions.R"))

#### Unzip and Load ####

quarter_files <- unzip(dials$zip_loc)

data_list <-
  quarter_files |>
  purrr::map(function(x) {
    
    files <- unzip(x) 
    
    loss <- get_zip_info(files, dials, dials$loss_cols, TRUE)
    orig <- get_zip_info(files, dials, dials$orig_cols, FALSE)
      
    loss_df <-
      data.table::fread(
        input = loss$path,
        sep = "|",
        header = FALSE,
        col.names = loss$names,
        colClasses = loss$types,
        select = loss$index,
        showProgress = TRUE
      ) |>
      tidyr::drop_na() |>
      dplyr::filter(
        zero_bal_code %in% c("02", "03", "09", "15"),
        net_loss != 0
      ) |>
      dplyr::group_by(id) |>
      dplyr::reframe(
        gross_loss = sum(
          delq_interest +
          zero_bal_upb -
          net_sale -
          expenses
        )
      ) |>
      dplyr::filter(
        gross_loss > 0
      ) |>
      dplyr::mutate(
        gross_loss = 1e4 * gross_loss / upb
      )
    
    quarter_df <-
      data.table::fread(
        input = orig$path,
        sep = "|",
        header = FALSE,
        col.names = orig$names,
        colClasses = orig$types,
        select = orig$index,
        showProgress = TRUE
      ) |>
      dplyr::inner_join(
        y = loss_df,
        by = "id"
      ) |>
      dplyr::select(- id)
    
    fs::file_delete(files)
    
    quarter_df
    
  }) |>
  dplyr::bind_rows() |>
  tibble::as_tibble() |>
  readr::write_rds(
    file = here::here("data/complete_data.Rds"),
    compress = "xz"
  )

fs::file_delete(quarter_files)
