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
  library(flashlight)
  library(ggplot2)
  library(scales)
  library(reticulate)
  library(probably)
  library(pROC)
  library(collapse)
  library(kableExtra)
  library(sjmisc)
})

TYPE <- Sys.getenv("TYPE") %>% as.integer()
print(paste("Working on TYPE ", TYPE))

MAX_WORDS <- 512L
MAXLEN <- 512L
DIM_SIZE <- 300L
options(filenamer.timestamp=1)
basedir <- "/research/dataset/AP_Developmental/RateMyProfessor/"

rmp <- readRDS("/research/dataset/AP_Developmental/RateMyProfessor/rmp.cleaned.coded.rds")

#model_file <- "/research/dataset/AP_Developmental/RateMyProfessor/model_output_2025-03-01.hdf5" 
model_file <- case_when(TYPE==1 ~ "/research/dataset/AP_Developmental/RateMyProfessor/model_output_1_2025-05-18.hdf5",
                        TYPE==2 ~ "/research/dataset/AP_Developmental/RateMyProfessor/model_output_2_2025-05-18.hdf5",
                        TRUE ~ "/research/dataset/AP_Developmental/RateMyProfessor/model_output_2025-03-27.hdf5")

#model_file <- "/research/dataset/AP_Developmental/RateMyProfessor/model_output_2025-03-27.hdf5" 


# extract the output
y_train = rmp[[1]] %>%
  mutate(ethnicity = as.factor(ethnicity)) %>%
  select(ethnicity) %>%
  as.data.table() %>%
  one_hot() %>%
  as.matrix()

y.col_sum <- colSums(y_train)
names(y.col_sum) <- 0:(ncol(y_train)-1)
class.weights <- (min(y.col_sum) / y.col_sum)

rm(y_train)
gc()

{
  df.train <- rmp[[1]] %>%
    select(-c("coded.professor_name")) %>%
    mutate(ethnicity = as.factor(ethnicity)) %>%
    slice(sample(1:n()))
  
  
  sklearn <- import("sklearn.utils")
  np <- import("numpy", convert=FALSE)
  compute_class_weights <- sklearn$compute_class_weight
  focal_loss_class_weights <- compute_class_weights(class_weight='balanced', 
                                                    classes=np$unique(df.train$ethnicity), 
                                                    y=np$array(df.train$ethnicity))
  
  weight_table <-  data.frame(table(df.train$ethnicity))
  weight_table$weights = focal_loss_class_weights
  sample_weight <- weight_table[df.train$ethnicity, "weights"]
  source("/research/SRC/AP_Developmental/loss_functions.R")
}

{
  CODED_CONSTRUCTS <- colnames(df.train) %>%
    setdiff(c("review_text", "ethnicity"))
  word_seqs.train = text_tokenizer(num_words = MAX_WORDS) %>%
    fit_text_tokenizer(df.train$review_text)
  x_train.text <- texts_to_sequences(word_seqs.train, df.train$review_text) %>%
    pad_sequences(maxlen = MAXLEN)
  x_train.constructs <- df.train[,CODED_CONSTRUCTS] %>%
    as.matrix()
}

df.test <- rmp[[2]] %>%
  #select(-c("coded.professor_name")) %>%
  mutate(ethnicity = as.factor(ethnicity))

sample_weight.test <- weight_table[df.test$ethnicity, "weights"]

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
  rm(word_seqs.train)
  gc()
}


ggplotConfusionMatrix <- function(m, my_f1, my_auc){
  mytitle <- paste("Accuracy", percent_format()(m$overall[1]),
                   ", Kappa", percent_format()(m$overall[2]), 
                   ", F1", percent_format()(my_f1),
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

custom_objects <- list(F1 = F1)
model_keras <- load_model_hdf5(model_file,  custom_objects=custom_objects)

y_hat.probs.test <- predict(model_keras, list(x_test.text, x_test.constructs),
                            batch_size = 1024L, type = "prob")
colnames(y_hat.probs.test) <- levels(df.test$ethnicity)

y_hat_probs.train <- predict(model_keras, list(x_train.text, x_train.constructs),
                             batch_size = 1024L, type = "prob")

to.save <- list(y.train = df.train$ethnicity,
                y_hat.train_probs = y_hat_probs.train,
                y.test = df.test$ethnicity,
                y_hat.test_probs = y_hat.probs.test)

to.save %>%
  saveRDS(paste0(basedir, "/for_cfm_", TYPE, ".rds"))

y_hat.test <- predict(model_keras, list(x_test.text, x_test.constructs),
                      batch_size = 1024L)
test_ethnicity_predicted <- as.character(y_hat.test)
colnames(y_hat.test) <- levels(df.test$ethnicity)
test_ethnicity_predicted <- colnames(y_hat.test)[apply(y_hat.test,1,which.max)] %>%
  factor(levels = levels(df.test$ethnicity))
table(test_ethnicity_predicted)

####
#Samplers
tdf <- data.frame(review_text = df.test$review_text,
                  y_hat.test,
                  coded_ethnicity = df.test$ethnicity,
                  predicted_ethnicity = test_ethnicity_predicted) %>%
  mutate(correct = (coded_ethnicity == predicted_ethnicity))

#case1
set.seed(1)
tdf %>% 
  sample_frac(1L) %>%
  filter(correct == TRUE) %>%
  filter(coded_ethnicity == "Black") %>%
  head(1)

set.seed(3)
tdf %>% 
  sample_frac(1L) %>%
  filter(correct == TRUE) %>%
  filter(coded_ethnicity == "White") %>%
  head(1)

set.seed(3)
tdf %>% 
  sample_frac(1L) %>%
  filter(correct == TRUE) %>%
  filter(coded_ethnicity == "Asian") %>%
  head(1)

set.seed(2)
tdf %>% 
  sample_frac(1L) %>%
  filter(correct == TRUE) %>%
  filter(coded_ethnicity == "Hispanic") %>%
  head(1)
####

py.f1 <- import("sklearn.metrics")$f1_score
py.conf <- import("sklearn.metrics")$confusion_matrix
py.report <- import("sklearn.metrics")$classification_report

py.report(y_true = df.test$ethnicity, y_pred = test_ethnicity_predicted, sample_weight = sample_weight.test)

test.f1 <- py.f1(y_true = df.test$ethnicity, y_pred = test_ethnicity_predicted, average = "weighted")
test.cfm <- py.conf(y_true = df.test$ethnicity, y_pred = test_ethnicity_predicted, sample_weight = sample_weight.test) %>%
  round()
test.auc <- multiclass.roc(df.test$ethnicity, y_hat.probs.test)
rownames(test.cfm) <- c("Asian", "Black", "Hispanic", "White")
colnames(test.cfm) <- c("Asian", "Black", "Hispanic", "White")
test.cfm

cfm <- confusionMatrix(table(reference = df.test$ethnicity, data = test_ethnicity_predicted))
cfm
{
  sink(paste0(basedir, "cfm_", TYPE, ".txt"))
  print(cfm)
  sink()
}
ggplotConfusionMatrix(cfm, test.f1, as.numeric(test.auc$auc))
ggsave(paste0(basedir, "cfm_", TYPE, ".jpg"), 
       width = 5.5, height = 2, units =  "in", dpi=300)

df.test$predicted_ethnicity <- test_ethnicity_predicted

df.test.profs <- df.test %>%
  group_by(coded.professor_name) %>%
  summarize(ethnicity = fmode(ethnicity),
            predicted_ethnicity = fmode(predicted_ethnicity))

confusionMatrix(reference = df.test.profs$ethnicity, data = df.test.profs$predicted_ethnicity)

if (FALSE) {
  fl_df <- data.frame(x_train.text, 
                      x_train.constructs, 
                      ethnicity = as.integer(df.train$ethnicity))
  fl <- flashlight(model = model_keras,
                   data = fl_df,
                   y = "ethnicity",
                   label = "roa",
                   predict_function = function(m,x) {
                     my.x_train.constructs <- x[,CODED_CONSTRUCTS] %>%
                       as.matrix()
                     my.x_train.text <- x[1:512] %>%
                       as.matrix()
                     my_y_hat <- predict(object = m, 
                                         list(my.x_train.text, my.x_train.constructs), 
                                         batch_size = 1024L,
                                         verbose = 0)
                     colnames(my_y_hat) <- levels(df.train$ethnicity)
                     my_ethnicity_predicted <- colnames(my_y_hat)[apply(my_y_hat,1,which.max)] %>%
                       factor(levels = levels(df.train$ethnicity)) %>%
                       as.integer()
                     return(my_ethnicity_predicted)
                   })
  imp <- light_importance(fl, m_repetitions = 10L)
  p <- plot(imp, top_m = 10, error_bars=FALSE)
}

if (TRUE) {
  df.train.all <- data.frame(x_train.text, x_train.constructs)
  df.train.all_y <- df.train[, "ethnicity"] %>%
    as.data.table() %>%
    one_hot() %>%
    as.matrix()
  explainer_keras_dalex <- DALEX::explain(model_keras,
                                          data = df.train.all, 
                                          y = df.train.all_y,
                                          label = "Overall model",
                                          predict_function = function(m,x) {
                                            my.x_train.constructs <- x[,CODED_CONSTRUCTS] %>%
                                              as.matrix()
                                            my.x_train.text <- x[1:512] %>%
                                              as.matrix()
                                            my_y_hat <- predict(object = m, 
                                                                list(my.x_train.text, my.x_train.constructs), 
                                                                batch_size = 1024L,
                                                                verbose = 0)
                                            colnames(my_y_hat) <- levels(df.train$ethnicity)
                                            return(my_y_hat)
                                          })
  
  interesting_coded_constructs <- setdiff(CODED_CONSTRUCTS, c("male", "female"))
  
  # use type = "raw" for raw plotting
  set.seed(101)
  vi_keras_dalex <- feature_importance(explainer_keras_dalex,
                                       type = "ratio",
                                       loss_function = loss_root_mean_square,
                                       variables= interesting_coded_constructs,
                                       B = 100)
  plot(vi_keras_dalex, max_vars = 10, show_boxplots = FALSE, bar_width = 6) +
    labs(title = NULL,
         subtitle = NULL,
         caption = NULL) +
    xlab("Top-20 coded constructs (ranked)") +
    ylab("Dropout loss") +
    theme_bw() +
    theme(legend.position = "none",
          plot.subtitle = element_blank())
  ggsave(paste0(basedir, "/vic.jpg"),
         height = 3, width = 5, units = "in", dpi = 300)
  
  top_vars <- vi_keras_dalex %>%
    as.data.frame() %>%
    arrange(desc(dropout_loss)) %>%
    filter(permutation == 0)
  
  writexl::write_xlsx(top_vars, paste0(basedir, "all_ranked_predictors.xlsx"))
  
  all_vars <- c(paste0("X", 1:512), CODED_CONSTRUCTS)
  names(all_vars) <- c(rep("GloVe Embeddings", 512), rep("Automatic coded constructs", length(CODED_CONSTRUCTS)))
  vars_list <- split(unname(all_vars),names(all_vars))              
  vi_grouped_keras_dalex <- feature_importance(explainer_keras_dalex,
                                               type = "ratio",
                                               loss_function = loss_root_mean_square,
                                               variable_groups = vars_list,
                                               B = 100)
  plot(vi_grouped_keras_dalex, show_boxplots = FALSE, bar_width = 6) +
    labs(title = NULL,
         subtitle = NULL,
         caption = NULL) +
    ylab("Dropout loss") +
    theme_bw() +
    theme(legend.position = "none",
          plot.subtitle = element_blank())
  ggsave(paste0(basedir, "/grouped_vic.jpg"),
         height = 1.4, width = 6, units = "in", dpi = 300)
  
  df.summary <- df.train %>%
    group_by(ethnicity) %>%
    summarize(M1 = round(mean(get(top_vars$variable[1])), 4),
              M2 = round(mean(get(top_vars$variable[2])), 4),
              M3 = round(mean(get(top_vars$variable[3])), 4),
              M4 = round(mean(get(top_vars$variable[4])), 4),
              M5 = round(mean(get(top_vars$variable[5])), 4),
              M6 = round(mean(get(top_vars$variable[6])), 4),
              M7 = round(mean(get(top_vars$variable[7])), 4),
              M8 = round(mean(get(top_vars$variable[8])), 4),
              M9 = round(mean(get(top_vars$variable[9])), 4),
              M10 = round(mean(get(top_vars$variable[10])), 4),
    )
  colnames(df.summary) <- c("Ethnicity", top_vars$variable[1:10])
  
  df.summary %>%
    kable(format = "latex",
          caption = "Top ten predictors and their average scores across ethnicities",
          label = "summary-ten",
          booktabs = TRUE,
          longtable = FALSE,
          linesep = "",
          align = "l",
          escape = TRUE) %>%
    kable_styling(position = "left",
                  full_width = F,
                  latex_options = c("scale_down", "HOLD_position")
    ) %>%
    writeLines(paste0(basedir, "summary-ten.tex"))
  
  
  all_ranked <- top_vars$variable %>%
    setdiff(c("_full_model_", "_baseline_"))
  
  score_variable <- function(myvar = NULL, tdf = NULL) {
    scored <- tdf %>%
      group_by(ethnicity) %>%
      summarize(M = round(mean(get(myvar)), 4)) %>%
      as.data.table()
    scored_t <- rotate_df(scored, cn = TRUE)
    rownames(scored_t) <- myvar
    return(scored_t)
  }
  
  all_scored <- lapply(all_ranked, score_variable, tdf = df.train) %>%
    do.call(rbind, .)
  
  all_scored2 <- data.frame(variables = all_ranked,
                            all_scored)
  rownames(all_scored2) <- paste0("#", 1:nrow(all_scored2))
  
  all_scored2 %>%
    kbl(format = "latex",
        caption = "All ranked predictors and their grouped scores",
        label = "summary-all",
        booktabs = TRUE,
        longtable = TRUE,
        linesep = "",
        align = "l",
        row.names = TRUE,
        escape = TRUE) %>%
    kable_styling(position = "left",
                  latex_options = c("striped", "repeat_header"),
                  font_size = 7) %>%
    writeLines(paste0(basedir, "summary-all.tex"))
  
  save.image(paste0(basedir, "/analysis.Rdata"))
}
