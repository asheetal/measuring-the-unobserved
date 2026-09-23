#!/usr/bin/Rscript

Sys.setenv("CUDA_VISIBLE_DEVICES"=0)
suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(data.table)
  library(mltools)
  library(word2vec)
  library(filenamer)
  library(keras)
  library(caret)
  library(wconf)
  library(ingredients)
  library(DALEX)
  library(ggplot2)
  library(scales)
  library(reticulate)
  library(probably)
  library(pROC)
})

MAX_WORDS <- 512L
MAXLEN <- 512L
DIM_SIZE <- 300L
options(filenamer.timestamp=1)
basedir <- "/research/dataset/AP_Developmental/RateMyProfessor/"
source("/research/SRC/AP_Developmental/loss_functions.R")

rds <- readRDS("/research/dataset/AP_Developmental/rateMDs/rmds.coded.names.rds") %>%
  filter(ethnicity %in% c("Asian", "Black", "Hispanic", "White")) %>%
  rename(review_text = text)

model_file <- "/research/dataset/AP_Developmental/RateMyProfessor/model_output_2025-02-15.hdf5"

df.test <- rds %>%
  select(-c("specialization", "doc_name", "country")) %>%
  filter(gender %in% c("Female", "Male")) %>%
  mutate(gender = as.factor(gender)) %>%
  as.data.table() %>%
  one_hot() %>%
  rename(male = gender_Male,
         female = gender_Female) %>%
  mutate(ethnicity = as.factor(ethnicity))

CODED_CONSTRUCTS <- setdiff(colnames(df.test), c("review_text", "ethnicity"))

{
  # tokenize the input data and then fit the created object
  word_seqs = text_tokenizer(num_words = MAX_WORDS) %>%
    fit_text_tokenizer(df.test$review_text)
  # apply tokenizer to the text and get indices instead of words
  # later pad the sequence
  x_test.text <- texts_to_sequences(word_seqs, df.test$review_text) %>%
    pad_sequences(maxlen = MAXLEN)
  x_test.constructs <- select(df.test, all_of(CODED_CONSTRUCTS)) %>%
    as.matrix()
  rm(word_seqs)
  gc()
}


ggplotConfusionMatrix <- function(m, my_f1, my_auc){
  mytitle <- paste("Accuracy", percent_format()(m$overall[1]),
                   ", Kappa", percent_format()(m$overall[2]), 
                   ", F1", round(my_f1, 3),
                   ", Multiclass AUC", round(my_auc, 3))
  p <-
    ggplot(data = as.data.frame(m$table) ,
           aes(x = reference, y = data)) +
    geom_tile(aes(fill = log(Freq)), colour = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(x = reference, y = data, label = Freq)) +
    theme(legend.position = "none", axis.title = element_text(size = 10), plot.title = element_text(size = 10)) +
    labs(x = "Reference", y = "Prediction") +
    ggtitle(mytitle)
  return(p)
}

custom_objects <- list(F1 = F1,
                       my_categorical_focal_loss = my_categorical_focal_loss,
                       my_categorical_hinge_loss = my_categorical_hinge_loss)
model_keras <- load_model_hdf5(model_file,  custom_objects=custom_objects)

y_hat.probs.test <- predict(model_keras, list(x_test.text, x_test.constructs),
                            batch_size = 1024L, type = "prob")
colnames(y_hat.probs.test) <- levels(df.test$ethnicity)

y_hat.test <- predict(model_keras, list(x_test.text, x_test.constructs),
                      batch_size = 1024L)

test_ethnicity_predicted <- as.character(y_hat.test)
colnames(y_hat.test) <- levels(df.test$ethnicity)
test_ethnicity_predicted <- colnames(y_hat.test)[apply(y_hat.test,1,which.max)] %>%
  factor(levels = levels(df.test$ethnicity))
table(test_ethnicity_predicted)

py.f1 <- import("sklearn.metrics")$f1_score
py.conf <- import("sklearn.metrics")$confusion_matrix
py.report <- import("sklearn.metrics")$classification_report

test.f1 <- py.f1(y_true = df.test$ethnicity, y_pred = test_ethnicity_predicted, average = "weighted")

test.auc <- multiclass.roc(df.test$ethnicity, y_hat.probs.test)

cfm <- confusionMatrix(table(reference = df.test$ethnicity, data = test_ethnicity_predicted))
cfm
{
  sink(paste0(basedir, "cfm_rmd.txt"))
  print(cfm)
  sink()
}
ggplotConfusionMatrix(cfm, test.f1, as.numeric(test.auc$auc))

ggsave(paste0(basedir, "cfm_rmd_raw.jpg"), 
       width = 5.5, height = 2, units =  "in", dpi=300)

list(y.test = df.test$ethnicity,
     y_hat.test_probs = y_hat.probs.test) %>%
  saveRDS(paste0(basedir, "/for_cfm_rmd.rds"))
