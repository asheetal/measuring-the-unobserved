#!/usr/bin/Rscript
Sys.setenv("CUDA_VISIBLE_DEVICES"=0)

library(tidyverse)
library(kableExtra)
library(sur)
library(fedmatch)
library(filenamer)

source("/research/SRC/AP_Developmental/text2code_functions.gpu.R")

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

all_professors.pre <- data.frame(professor_name = prof_names,
                                 ethnicity_raw = add_ethnicity(data.frame(text = prof_names)),
                                 add_gender(data.frame(text = prof_names)))
all_professors.pre <- data.frame(professor_name = prof_names,
                             ethnicity_raw = add_ethnicity(data.frame(text = prof_names)),
                             add_gender(data.frame(text = prof_names)))
#report percentages
x <- all_professors.pre %>%
  select(-c("professor_name", "male", "female")) %>%
  rowwise() %>%
  mutate(Max_Col = names(.)[which.max(c_across(everything()))]) %>%
  ungroup()
percent.table(x$Max_Col)


all_professors.ethnicity <- all_professors.pre %>%
  select(starts_with("ethnicity")) %>%
  setNames(gsub("ethnicity_raw.","",names(.))) %>%
  mutate(White = nordic + french + italian + germanic + east_european + jewish,
         Asian = south_asian + east_asian + japanese,
         Other = british, #British names can be blacks as well, ambiguous
         Hispanic = hispanic,
         Black = african_muslim + african_africans) %>%
  select(c("Asian", "Black", "Hispanic", "White", "Other"))
ethnicity_coded <- colnames(all_professors.ethnicity)[max.col(all_professors.ethnicity)]

all_professors <- data.frame(professor_name = all_professors.pre$professor_name,
                             male = all_professors.pre$male,
                             female = all_professors.pre$female,
                             ethnicity = ethnicity_coded)
  
df.coded.ethnicity <- df %>%
  left_join(all_professors, by = c("professor_name" = "professor_name"))

#report final instructors minus Other
df.coded.ethnicity %>%
  filter(ethnicity != "Other") %>%
  mutate(professor_name = clean_strings(professor_name)) %>%
  group_by(professor_name) %>%
  dplyr::summarise(ethnicity = head(ethnicity, 1),
            male = round(head(male, 1)),
            female = 1 - male,
            reviews = n()) %>%
  group_by(ethnicity) %>%
  summarise(N = n(),
            male = sum(male),
            female = sum(female),
            reviews = sum(reviews))


#save_file <- "/research/dataset/AP_Developmental/RateMyProfessor/rmp.coded.all.rds"
options(filenamer.timestamp=1)
save_filename <- filename("rmp.coded.all",
                          path="/research/dataset/AP_Developmental/RateMyProfessor",
                          tag=NULL,
                          ext="rds",
                          subdir=FALSE) %>%
  as.character() %>%
  print()

#df.coded.ethnicity <- readRDS(save_file)
df.coded.all <- df.coded.ethnicity %>%
  mutate(text = review_text) %>%
  my_add_all_codes(save_file)

saveRDS(df.coded.ethnicity, save_filename)


