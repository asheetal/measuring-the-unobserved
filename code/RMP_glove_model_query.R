#!/usr/bin/Rscript
suppressPackageStartupMessages({
  library(xlsx2dfs)
  library(word2vec)
  library(text2vec)
  library(tm)
  library(slam)
  library(reticulate)
  library(stringi)
  library(stringr)
  library(dplyr)
  library(lsa)
  library(ggplot2)
  library(ggpubr)
  library(parallel)
  library(broom)
  library(readxl)
  library(writexl)
  library(janitor)
  library(lme4)
  library(foreign)
  library(data.table)
  library(purrr)
  library(scales)
  library(textdata)
  library(tidyr)
  library(scales)
  library(kableExtra)
})
base_model_file <- "/research/dataset/GLOVE/glove.42B.300d_gensim.txt"
vad_words <- lexicon_nrc_vad()

#RateMyProfessor
base_dir <- '/research/dataset/AP_Developmental/RateMyProfessor/'

gensim <- import('gensim.models')
KeyedVectors <- gensim$KeyedVectors

#load all yearly finetuned models
model_files <- paste0(base_dir, "/word2vec_", c("asian", "black", "hispanic", "white"), ".42B.model")
year_glove_models <- lapply(model_files, function(x) {return(KeyedVectors$load_word2vec_format(x, binary=T))})
year_glove_models_vocab <- lapply(year_glove_models, function(x) {return(names(x$index_to_key))})

# Load the Stanford GLOVE model
baseline_glove_model <- KeyedVectors$load_word2vec_format(base_model_file, binary=F)
#baseline_glove_vocab <- names(baseline_glove_model$index_to_key)
baseline_glove_vocab <- names(baseline_glove_model$index2word) #Gensim 3


instructor_synonyms <- c("professor", "teacher", "mentor", "instructor", "tutor", "lecturer", "guide", "advisor", "educator", "scholar")

document_vector <- function(doc, mod, vocab) {
  doc_words <- doc %>%
    stri_enc_toutf8() %>%
    tolower() %>%
    sample() %>%
    str_squish() %>%
    str_replace_all("[^[:alnum:][:space:]]", "") %>%
    strsplit(split = " ") %>%
    unlist()
  # Find which words are in-vocabulary (assuming rownames are the words in the glove_model)
  in_vocab <- doc_words %in% vocab
  my_doc <- doc_words[in_vocab]
  
  my_get_vec <- function(my_word) {
    return(mod$wv[my_word])
  }
  # Get the vectors for the in-vocabulary words
  word_vectors <- sapply(my_doc, my_get_vec)
  
  if (length(my_doc) == 0) {
    return(rep(NA, mod$vector_size)) # return an NA vector if no words match
  }
  
  # Calculate the mean vector (column-wise mean)
  mean_vector <- rowMeans(word_vectors, na.rm = TRUE)
  return(mean_vector)
}

topic_relevance_score_wrt_base <- function(year_id, my_topic) {
  my_topic_vector <- document_vector(my_topic, year_glove_models[[year_id]], year_glove_models_vocab[[year_id]])
  
  my_topic_vector_base <- document_vector(my_topic, baseline_glove_model, baseline_glove_vocab)
  cosine_similarities <- cosine(my_topic_vector, my_topic_vector_base)
  return(data.frame(cosine_similarities = cosine_similarities))
}


topic_trend <- function(topic) {
  trends <- lapply(1:length(year_glove_models), topic_relevance_score_wrt_base, my_topic = topic)
  trends_df <- do.call(rbind, trends) %>%
    cbind(year = 1:length(year_glove_models))
  return(trends_df)
}

top_five_similar <- function(mod = NULL, my_word = NULL) {
  x <- do.call(rbind, mod$similar_by_word(word = my_word, topn = 5L))[,1] %>% unlist()
  return(x)
}

my_word_distance <- function(words = NULL, all_mods = NULL) {
  x <- lapply(all_mods, function(mod, word1, word2) { mod$distance(word1, word2)}, word1 = words[1], word2 = words[2]) %>%
    unlist()
  return(x)
}

my_plot <- function(i = NULL, my_word_pairs = NULL, my_values = NULL) {
  
  my_word_pair <- my_word_pairs[[i]]
  value <- my_values[[i]]
  
  min_value = min(value)
  max_value = ifelse(1/max(value) > 10, max(value) + (max(value)/10), 1.0)
  my_text <- paste(my_word_pair, collapse = " & ")
  my_df <- data.frame(Ethnicity = c("Asian", "Black", "Hispanic", "White"),
                      CosineDist = value[2:5])
  p <- my_df %>%
    mutate(Ethnicity = as.factor(Ethnicity)) %>%
    #ggplot(aes(x=reorder(Ethnicity, CosineDist), y = CosineDist, fill=as.factor(Ethnicity) )) + 
    ggplot(aes(x=Ethnicity, y = CosineDist, fill=as.factor(Ethnicity) )) + 
    geom_bar(stat = "identity") +
    geom_abline(slope=0, intercept=value[1],  col = "black",lty=2) +
    scale_fill_brewer(palette = "Set1") +
    labs(x = my_text, y = NULL, fill = "Ethnicity") +
    scale_y_continuous(limits = c((min_value - min_value/10), 1.0), oob=rescale_none) +
    #scale_y_reverse() +
    theme_minimal() +
    theme(legend.position="none",
          axis.title.x = element_text(size=7),
          axis.text.x=element_blank(),
          axix.text.y=element_blank())
  return(p)
}

if (FALSE) {
  all_models <- c(baseline_glove_model, year_glove_models)
  word_pairs <- list(
    c("effort", "ability"),
    c("intimacy", "vulnerability"),
    c("trust", "respect"),
    c("warmth", "competence"),
    c("support", "autonomy"),
    c("fairness", "empathy"),
    c("authority", "submission"),
    c("expectation", "achievement"),
    c("feedback", "receptivity"),
    c("conflict", "resolution"),
    c("engagement", "motivation"),
    c("discipline", "responsiveness"),
    c("interest", "encouragement"),
    c("communication", "understanding"),
    c("involvement", "recognition"),
    c("caring", "demanding"),
    c("control", "independence")
  )
  #lapply(all_models, top_five_similar, my_word = "growth")
  values <- lapply(word_pairs, my_word_distance, all_mods = all_models)
  plot_list <- lapply(1:length(word_pairs), my_plot, my_word_pairs = word_pairs, my_values = values)
  ggarrange(plotlist = plot_list, ncol = 3, nrow = 5, legend = "bottom", common.legend = TRUE)
  ggsave(paste0(base_dir, "cosine_distance_comparison.jpg"),
         height = 11, width = 6.5, dpi = 300, units = "in")
  
  word_pairs_df <- lapply(word_pairs, paste0, collapse = " & ")
  values_df <- do.call(rbind, values) %>% round(4)
  colnames(values_df) <- c("Baseline", "Asian", "Black", "Hispanic", "White")
  tdf <- data.frame(`Construct pairs` = unlist(word_pairs_df),
                    values_df)
  tdf %>%
    kbl(format = "latex",
        caption = "Effect of contextualizing the word embedding models",
        label = "word-embedding-effect",
        booktabs = TRUE,
        longtable = FALSE,
        linesep = "",
        align = "l",
        escape = TRUE) %>%
    column_spec(2:6, width = "6em") %>%
    kable_styling(position = "left",
                  latex_options = c("striped", "repeat_header"),
                  font_size = 7) %>%
    writeLines('/research/dataset/AP_Developmental/RateMyProfessor/word-embedding-contexts-effect.tex')
}


my_model_vad <- function(my_word = NULL, model = NULL) {
  a <- rbindlist(model$similar_by_word(my_word, topn = length(model$key_to_index))) %>%
    filter(V1 %in% vad_words$Word) %>%
    left_join(vad_words, by = c("V1" = "Word")) %>%
    head(200)
  return(a)
}

my_model_var_score <- function(model = NULL, anchor_words = NULL) {
  a <- lapply(anchor_words, my_model_vad, model = model)
  b <- do.call(rbind, a)
  
  V <- weighted.mean(b$Valence, b$V2)
  A <- weighted.mean(b$Arousal, b$V2)
  D <- weighted.mean(b$Dominance, b$V2)
  
  data.frame(Valence = V, Arousal = A, Dominance = D) %>%
    return()
}

all_vads <- lapply(c(year_glove_models, baseline_glove_model), my_model_var_score, instructor_synonyms) %>%
  do.call(rbind, .)

all_vads$Group <- c("Asian", "Black", "Hispanic", "White", "Baseline")

all_vads2 <- all_vads %>%
  head(4) %>%
  mutate_at(c("Valence", "Arousal", "Dominance"), ~(rescale(.) %>% as.vector))

df_long_unscaled <- all_vads %>%
  head(4) %>%
  pivot_longer(cols = -Group, names_to = "Variable", values_to = "Score2")

df_long <- all_vads2 %>%
  pivot_longer(cols = -Group, names_to = "Variable", values_to = "Score")
df_long$Score2 = df_long_unscaled$Score2

df_long <- mutate(df_long, Variable = factor(Variable, levels = c("Valence", "Arousal", "Dominance")))

ggplot(df_long, aes(x = Variable, y = Group, fill = Score)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.3f", Score)), color = "black", size = 4) +  # Adds text to tiles
  scale_fill_gradient2(low = "red", high = "#CC00FF", mid = "green", midpoint = mean(df_long$Score),
                       transform = "log1p", limit = c(min(df_long$Score), max(df_long$Score))) +
  theme_minimal() +
  labs(x = "Variable", y = "Group") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position="none")

ggsave("/research/dataset/AP_Developmental/RateMyProfessor/VAD_heatmap.jpg",
       height = 3, width = 4, units = "in", dpi = 300)
