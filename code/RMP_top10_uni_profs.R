#!/usr/bin/Rscript

library(tidyverse)
library(kableExtra)
library(sur)
library(fedmatch)
library(filenamer)

jsons <- "/research/dataset/AP_Developmental/RateMyProfessor/final_data/"
temp <- list.files(jsons, pattern="*.json", full.names=TRUE)

all_unis <- lapply(temp, jsonlite::read_json)

get_all_reviews_faris <- function(l = NULL) {
  n <- length(l)
  if (n == 0) {
    return(NA)
  } else {
    all_faris_reviews <- sapply(l, function(x) {x[["review_text"]]})
    return(all_faris_reviews)
  }
}

get_all_professors_reviews <- function(my_uni = NULL) {
  n_profs <- length(my_uni)
  all_prof_reviews_list <- lapply(my_uni, function(x) {data.frame(professor_name = x$professors[[1]]$name,
                                                                  review_text = get_all_reviews_faris(x$professors[[1]]$student_reviews),
                                                                  quality = x$professors[[1]]$quality,
                                                                  difficulty = x$professors[[1]]$difficulty,
                                                                  reenroll = x$professors[[1]]$would_take_again / x$professors[[1]]$ratings_count)})
  all_prof_reviews_df <- do.call(rbind, all_prof_reviews_list)
  all_prof_reviews_df$university <- my_uni[[1]]$university
  return(all_prof_reviews_df)
}

get_combined_usa_unis_reviews <- function(my_all_unis = NULL) {
  n_unis <- length(my_all_unis)
  all_usa_reviews <- lapply(my_all_unis, get_all_professors_reviews)
  all_usa_reviews_df <- do.call(rbind, all_usa_reviews)
  return(all_usa_reviews_df)
}

df <- get_combined_usa_unis_reviews(all_unis)

df %>% 
  rename(University = university) %>%
  group_by(University) %>%
  summarize(`Number of professors` = length(unique(professor_name)),
            `Number of reviews` = n()) %>%
  arrange(desc(`Number of professors`)) %>%
  kable(format = 'latex', booktabs = FALSE,
        caption = "Raw numbers from 10 universities in the dataset",
        label = "crosstab-1") %>%
  kable_styling(bootstrap_options = c("striped")) %>%
  writeLines('/research/dataset/AP_Developmental/RateMyProfessor/crosstab-1.tex')


prof_names <- unique(df$professor_name)

all_professors.pre <- data.frame(professor_name = prof_names)

options(filenamer.timestamp=1)
save_filename <- filename("top10_uni_profs",
                          path="/research/dataset/AP_Developmental/RateMyProfessor",
                          tag=NULL,
                          ext="rds",
                          subdir=FALSE) %>%
  as.character(sanitize=FALSE) %>%
  print()

saveRDS(all_professors.pre, save_filename)
