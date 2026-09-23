library(tidyverse)
library(jsonlite)

rmp.coded.names <- readRDS("/research/dataset/AP_Developmental/RateMyProfessor/rmp.coded.names.rds")

unique_profs <- data.frame(professor_name = unique(rmp.coded.names$professor_name),
                           coded.professor_name = paste0("PROF", 1:length(unique(rmp.coded.names$professor_name))))

df <- rmp.coded.names %>%
  left_join(unique_profs, by = c("professor_name" = "professor_name")) %>%
  select(-c("professor_name", "university")) %>%
  rename(review_text = text) %>% 
  filter(ethnicity != "Other") %>%
  slice(sample(1:n()))

set.seed(3456)
n <- nrow(unique_profs)
trainIndex <- sample(1:n, size = round(0.9*n), replace=FALSE)
gvkey.train <- unique_profs[trainIndex, ]
gvkey.test <- unique_profs[-trainIndex, ]

df.train <- df %>%
  filter(coded.professor_name %in% gvkey.train$coded.professor_name)

df.test <- df %>%
  filter(coded.professor_name %in% gvkey.test$coded.professor_name)

df.train.groups <- df.train %>%
  #filter(ethnicity %in% c("Asian", "Black", "Hispanic", "White")) %>%
  filter(nchar(review_text) > 10) %>%
  group_by(ethnicity)

df.split.groups <- group_split(df.train.groups)
group_keys(df.train.groups)

names(df.split.groups) <- as.character(group_keys(df.train.groups)$ethnicity)

write_json(df.split.groups, "/research/dataset/AP_Developmental/RateMyProfessor/for_glove_splitted.json")
list(train = df.train,
     test = df.test) %>%
  saveRDS("/research/dataset/AP_Developmental/RateMyProfessor/rmp.cleaned.coded.rds")
