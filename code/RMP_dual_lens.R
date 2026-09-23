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
  library(collapse)
  library(parallel)
  library(tokenizers)
  library(textdata)
  library(kableExtra)
})

`%notin%` <- Negate(`%in%`)
MAX_WORDS <- 512L
MAXLEN <- 512L
DIM_SIZE <- 300L


basedir <- '/research/dataset/AP_Developmental/RateMyProfessor/'
rmp <- readRDS("/research/dataset/AP_Developmental/RateMyProfessor/rmp.cleaned.coded.rds")

df <- rbind(rmp[["train"]],
            rmp[["test"]])


vad_words <- lexicon_nrc_vad()

gensim <- import('gensim.models')
KeyedVectors <- gensim$KeyedVectors

#load all yearly finetuned models
model_files <- paste0(basedir, "/word2vec_", c("asian", "black", "hispanic", "white"), ".42B.model")
year_glove_models <- lapply(model_files, function(x) {return(KeyedVectors$load_word2vec_format(x, binary=T))})
year_glove_models_vocab <- lapply(year_glove_models, function(x) {return(names(x$index_to_key))})

#df.filtered <- df %>%
#  filter(ethnicity %in% c("Asian", "White"))

df.filtered <- df

word_vad <- function(word = NULL) {
  if (word %notin% vad_words$Word) {
    return(data.frame(V = NA, A = NA, D = NA))
  } else {
    vad <- vad_words[which(vad_words$Word == word), ]
    V <- head(as.numeric(vad$Valence), 1)
    A <- head(as.numeric(vad$Arousal), 1)
    D <- head(as.numeric(vad$Dominance), 1)
    return(data.frame(V = V, A = A, D = D))
  }
}

sentence_vad <- function(sentence = NULL) {
  word_seqs <- tokenize_words(sentence) %>% unlist()
  if (is_empty(word_seqs)) {
    ret <- data.frame(V = NA, A = NA, D = NA)
  } else {
    vad <- lapply(word_seqs, word_vad) %>% do.call(rbind, .) %>%
      remove_empty("rows")
    if (nrow(vad) > 0) {
      x <- colMeans(vad, na.rm = TRUE)
      ret <- data.frame(V = x["V"], A = x["A"], D = x["D"])
    } else {
      ret <- data.frame(V = NA, A = NA, D = NA)
    }
  }
  return(ret)
}

numcores <- detectCores() - 8
vad_scores <- mclapply(df.filtered$review_text, sentence_vad, mc.cores = numcores) %>%
  do.call(rbind, .)

save.image(paste0(basedir, "/dual_lens.RData"))
#load(paste0(basedir, "/dual_lens.RData"))

df.vadded <- cbind(df.filtered, vad_scores)

df.asian <- df.vadded %>%
  filter(ethnicity == "Asian")

df.black <- df.vadded %>%
  filter(ethnicity == "Black")

df.hispanic <- df.vadded %>%
  filter(ethnicity == "Hispanic")

df.white <- df.vadded %>%
  filter(ethnicity == "White")


model_file <- "/research/dataset/AP_Developmental/RateMyProfessor/model_output_2025-03-27.hdf5" 
source("/research/SRC/AP_Developmental/loss_functions.R")
custom_objects <- list(F1 = F1)
model_keras <- load_model_hdf5(model_file,  custom_objects=custom_objects)


get_vic <- function(dft = NULL, label = NULL){
  {
    CODED_CONSTRUCTS <- colnames(dft) %>%
      setdiff(c("review_text", "coded.professor_name", "ethnicity", "V", "A", "D"))
    word_seqs.train = text_tokenizer(num_words = MAX_WORDS) %>%
      fit_text_tokenizer(dft$review_text)
    x_train.text <- texts_to_sequences(word_seqs.train, dft$review_text) %>%
      pad_sequences(maxlen = MAXLEN)
    x_train.constructs <- dft[,CODED_CONSTRUCTS] %>%
      as.matrix()
    y_train = dft %>%
      mutate(ethnicity = factor(ethnicity, levels = c("Asian", "Black", "Hispanic", "White"))) %>%
      select(ethnicity) %>%
      as.data.table() %>%
      one_hot() %>%
      as.matrix()
  }
  
  df.train.all <- data.frame(x_train.text, x_train.constructs)
  explainer_keras_dalex <- DALEX::explain(model_keras,
                                          data = df.train.all, 
                                          y = as.numeric(y_train),
                                          label = label,
                                          predict_function = function(m,x) {
                                            my.x_train.constructs <- x[,CODED_CONSTRUCTS] %>%
                                              as.matrix()
                                            my.x_train.text <- x[1:512] %>%
                                              as.matrix()
                                            my_y_hat <- predict(object = m, 
                                                                list(my.x_train.text, my.x_train.constructs), 
                                                                batch_size = 1024L,
                                                                verbose = 0)
                                            as.numeric(my_y_hat) %>%
                                              return()
                                          })
  
  interesting_coded_constructs <- setdiff(CODED_CONSTRUCTS, c("male", "female"))
  
  # use type = "raw" for raw plotting
  vi_keras_dalex <- feature_importance(explainer_keras_dalex,
                                       type = "ratio",
                                       loss_function = loss_root_mean_square,
                                       variables= interesting_coded_constructs,
                                       B = 10)
  plot(vi_keras_dalex, max_vars = 10, show_boxplots = FALSE, bar_width = 6) +
    labs(title = NULL,
         subtitle = NULL,
         caption = NULL) +
    xlab("Top-20 coded constructs (ranked)") +
    ylab("Dropout loss") +
    theme_bw() +
    theme(legend.position = "none",
          plot.subtitle = element_blank())
  ggsave(paste0(basedir, "/", label, ".jpg"),
         height = 5, width = 5, units = "in", dpi = 300)
  
  top_vars <- vi_keras_dalex %>%
    as.data.frame() %>%
    arrange(desc(dropout_loss)) %>%
    filter(permutation == 0)
  df.summary <- dft %>%
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
              M10 = round(mean(get(top_vars$variable[10])), 4)
    )
  colnames(df.summary) <- c("Ethnicity", top_vars$variable[1:10])
  df.summary %>%
    kable(format = "latex",
        table.envir = "subtable",
        caption = label,
        label = gsub(" ", "-", label),
        booktabs = TRUE,
        longtable = FALSE,
        linesep = "",
        align = "l",
        escape = TRUE) %>%
    kable_styling(position = "left",
                  full_width = F,
                  latex_options = c("scale_down")
                  ) %>%
    writeLines(paste0(basedir, "summary_", label, ".tex"))
  #write.csv(df.summary, paste0(basedir, "/summary_", label, ".csv"), row.names = F)
}

##############################################

get_top_X <- function(dft = NULL, X = NULL) {
  return(filter(dft, get(X) > 0.80))
}

get_bottom_X <- function(dft = NULL, X = NULL) {
  return(filter(dft, get(X) < 0.20))
}

dft <- rbind(get_top_X(df.asian, "V"), 
             get_top_X(df.black, "V"),
             get_top_X(df.hispanic, "V"),
             get_top_X(df.white, "V")) %>%
  slice(sample(1:n()))
get_vic(dft, "Top quintile valence predictors")

dft <- rbind(get_bottom_X(df.asian, "V"),
             get_bottom_X(df.black, "V"),
             get_bottom_X(df.hispanic, "V"),
             get_bottom_X(df.white, "V")) %>%
  slice(sample(1:n()))
get_vic(dft, "Bottom quintile valence predictors")

dft <- rbind(get_top_X(df.asian, "A"), 
             get_top_X(df.black, "A"),
             get_top_X(df.hispanic, "A"),
             get_top_X(df.white, "A")) %>%
  slice(sample(1:n()))
get_vic(dft, "Top quintile arousal predictors")

dft <- rbind(get_bottom_X(df.asian, "A"),
             get_bottom_X(df.black, "A"),
             get_bottom_X(df.hispanic, "A"),
             get_bottom_X(df.white, "A")) %>%
  slice(sample(1:n()))
get_vic(dft, "Bottom quintile arousal predictors")

dft <- rbind(get_top_X(df.asian, "D"), 
             get_top_X(df.black, "D"),
             get_top_X(df.hispanic, "D"),
             get_top_X(df.white, "D")) %>%
  slice(sample(1:n()))
get_vic(dft, "Top quintile dominance predictors")

dft <- rbind(get_bottom_X(df.asian, "D"),
             get_bottom_X(df.black, "D"),
             get_bottom_X(df.hispanic, "D"),
             get_bottom_X(df.white, "D")) %>%
  slice(sample(1:n()))
get_vic(dft, "Bottom quintile dominance predictors")

##############################################3

get_VIC_equal_ratings <- function(dft = NULL, label = NULL){
  {
    CODED_CONSTRUCTS <- colnames(dft) %>%
      setdiff(c("review_text", "text", "university", "professor_name", "coded.professor_name", "ethnicity", "V", "A", "D",
                "quality", "difficulty", "reenroll"))
    word_seqs.train = text_tokenizer(num_words = MAX_WORDS) %>%
      fit_text_tokenizer(dft$review_text)
    x_train.text <- texts_to_sequences(word_seqs.train, dft$review_text) %>%
      pad_sequences(maxlen = MAXLEN)
    x_train.constructs <- dft[,CODED_CONSTRUCTS] %>%
      as.matrix()
    y_train = dft %>%
      mutate(ethnicity = factor(ethnicity, levels = c("Asian", "Black", "Hispanic", "White"))) %>%
      select(ethnicity) %>%
      as.data.table() %>%
      one_hot() %>%
      as.matrix()
  }
  
  df.train.all <- data.frame(x_train.text, x_train.constructs)
  explainer_keras_dalex <- DALEX::explain(model_keras,
                                          data = df.train.all, 
                                          y = as.numeric(y_train),
                                          label = label,
                                          predict_function = function(m,x) {
                                            my.x_train.constructs <- x[,CODED_CONSTRUCTS] %>%
                                              as.matrix()
                                            my.x_train.text <- x[1:512] %>%
                                              as.matrix()
                                            my_y_hat <- predict(object = m, 
                                                                list(my.x_train.text, my.x_train.constructs), 
                                                                batch_size = 1024L,
                                                                verbose = 0)
                                            as.numeric(my_y_hat) %>%
                                              return()
                                          })
  
  interesting_coded_constructs <- setdiff(CODED_CONSTRUCTS, c("male", "female"))
  
  # use type = "raw" for raw plotting
  vi_keras_dalex <- feature_importance(explainer_keras_dalex,
                                       type = "ratio",
                                       loss_function = loss_root_mean_square,
                                       variables= interesting_coded_constructs,
                                       B = 10)
  plot(vi_keras_dalex, max_vars = 20, show_boxplots = FALSE, bar_width = 6) +
    labs(title = NULL,
         subtitle = NULL,
         caption = NULL) +
    xlab("Top-20 coded constructs (ranked)") +
    ylab("Dropout loss") +
    theme_bw() +
    theme(legend.position = "none",
          plot.subtitle = element_blank())
  ggsave(paste0(basedir, "/", label, ".jpg"),
         height = 5, width = 5, units = "in", dpi = 300)
  
  top_vars <- vi_keras_dalex %>%
    as.data.frame() %>%
    arrange(desc(dropout_loss)) %>%
    filter(permutation == 0)
  df.summary <- dft %>%
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
              M10 = round(mean(get(top_vars$variable[10])), 4)
    )
  colnames(df.summary) <- c("Ethnicity", top_vars$variable[1:10])
  df.summary %>%
    kable(format = "latex",
          table.envir = "subtable",
          caption = label,
          label = gsub(" ", "-", label),
          booktabs = TRUE,
          longtable = FALSE,
          linesep = "",
          align = "l",
          escape = TRUE) %>%
    kable_styling(position = "left",
                  full_width = F,
                  latex_options = c("scale_down")
    ) %>%
    writeLines(paste0(basedir, "summary_", label, ".tex"))
}


readRDS("/research/dataset/AP_Developmental/RateMyProfessor/for_vic_top_quality.rds") %>%
  get_VIC_equal_ratings("High quality")
readRDS("/research/dataset/AP_Developmental/RateMyProfessor/for_vic_mid_quality.rds") %>%
  get_VIC_equal_ratings("Medium quality")
readRDS("/research/dataset/AP_Developmental/RateMyProfessor/for_vic_bottom_quality.rds") %>%
  get_VIC_equal_ratings("Low quality")
readRDS("/research/dataset/AP_Developmental/RateMyProfessor/for_vic_easiest.rds") %>%
  get_VIC_equal_ratings("Low difficulty")
readRDS("/research/dataset/AP_Developmental/RateMyProfessor/for_vic_mid_difficult.rds") %>%
  get_VIC_equal_ratings("Medium difficulty")
readRDS("/research/dataset/AP_Developmental/RateMyProfessor/for_vic_difficult.rds") %>%
  get_VIC_equal_ratings("High difficulty")
