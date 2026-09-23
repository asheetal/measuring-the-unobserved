suppressPackageStartupMessages({
  library(tidyverse)
  library(kableExtra)
  library(schmitz)
  library(Hmisc)
  library(xlsx)
  library(nnet)
  library(VGLM)
  library(car)
  library(textdata)
  library(tokenizers)
  library(janitor)
  library(parallel)
})
`%notin%` <- Negate(`%in%`)

rmp.cleaned.coded <- readRDS("/research/dataset/AP_Developmental/RateMyProfessor/rmp.cleaned.coded.rds")
df <- rbind(rmp.cleaned.coded[["train"]], rmp.cleaned.coded[["test"]])

df.n <- df %>%
  mutate(gender = ifelse(male > female, "M", "F")) %>%
  group_by(coded.professor_name) %>% 
  summarise(N = n(), 
            gender = head(gender, 1),
            ethnicity = head(ethnicity, 1))

ggplot(data=df.n, aes(x=N, group=gender, fill=gender)) +
  geom_density(adjust=1.5, alpha=.4) +
  scale_x_continuous(trans='log10') +
  labs(x = "number of reviews (log10 scale)") +
  theme_minimal()

ggsave("/research/dataset/AP_Developmental/RateMyProfessor/review_density.jpg",
       height = 5, width = 5, units = "in", dpi = 300)  

df.n %>%
  group_by(ethnicity, gender) %>%
  mutate(N = n()) %>%
  ggplot(aes(fill=gender, y=N, x=ethnicity)) + 
  geom_bar(position="dodge", stat="identity") +
  labs(y = "Number of instructors") +
  theme_minimal()

ggsave("/research/dataset/AP_Developmental/RateMyProfessor/ethnicity_barplot.jpg",
       height = 5, width = 5, units = "in", dpi = 300) 

df %>%
  mutate(gender = ifelse(male > female, "M", "F")) %>%
  group_by(coded.professor_name) %>% 
  summarise(N = n(),
            gender = head(gender, 1),
            ethnicity = head(ethnicity, 1)) %>%
  group_by(ethnicity, gender) %>%
  summarise(rate = mean(N)) %>%
  ggplot(aes(fill=gender, y=rate, x=ethnicity)) + 
  geom_bar(position="dodge", stat="identity") +
  labs(y = "Average number of reviews per instructor") +
  theme_minimal()

ggsave("/research/dataset/AP_Developmental/RateMyProfessor/average_reviews.jpg",
       height = 5, width = 5, units = "in", dpi = 300) 

df.stats <- df %>%
  group_by(professor_name) %>%
  summarise(quality = mean(quality, na.rm = T),
            difficulty = mean(difficulty, na.rm = T),
            #reenroll = mean(reenroll, na.rm = T),
            ethnicity = head(ethnicity, 1)) %>%
  select(-c("professor_name")) %>%
  filter(ethnicity != "Other") %>%
  group_by(ethnicity) %>%
  summarise(quality.mean = mean(quality),
            quality.lowerCL = t.test(quality)$conf.int[1],
            quality.upperCL = t.test(quality)$conf.int[2],
            difficulty.mean = mean(difficulty),
            difficulty.lowerCL = t.test(difficulty)$conf.int[1],
            difficulty.upperCL = t.test(difficulty)$conf.int[2])

df.means <- df %>%
  group_by(professor_name) %>%
  summarise(quality = mean(quality),
            difficulty = mean(difficulty))

#Forest plot
a <- ggplot(df.stats, aes(x=quality.mean, y=ethnicity, color=ethnicity, shape=ethnicity)) +
  geom_errorbar(aes(xmin = quality.lowerCL, xmax = quality.upperCL), width = 0.5) +
  geom_point(size = 4) + 
  labs(x="Mean quality rating and 95% CI (0 to 5)", y = "Ethnicity") +
  #scale_x_continuous(breaks=seq(-20,80,20), limits = c(-20,80)) +
  #xlim(-10, 70) + 
  #scale_color_manual(values = c("#4e8ca5", "#ff6d6d", "#66aa66", "#aa9966")) +
  scale_shape_manual(values=c(15,16,17,18)) +
  geom_vline(xintercept = mean(df.means$quality), linetype = "longdash") +
  theme_classic() +
  theme(legend.position = "none")

ggsave("/research/dataset/AP_Developmental/RateMyProfessor/average_quality.jpg",
       height = 3, width = 5, units = "in", dpi = 300, plot = a) 

b <- ggplot(df.stats, aes(x=difficulty.mean, y=ethnicity, color=ethnicity, shape=ethnicity)) +
  geom_errorbar(aes(xmin = difficulty.lowerCL, xmax = difficulty.upperCL), width = 0.5) +
  geom_point(size = 4) + 
  labs(x="Mean difficulty rating and 95% CI (0 to 5)", y = "Ethnicity") +
  #scale_x_continuous(breaks=seq(-20,80,20), limits = c(-20,80)) +
  #xlim(-10, 70) + 
  #scale_color_manual(values = c("#4e8ca5", "#ff6d6d", "#66aa66", "#aa9966")) +
  scale_shape_manual(values=c(15,16,17,18)) +
  geom_vline(xintercept = mean(df.means$difficulty), linetype = "longdash") +
  theme_classic() +
  theme(legend.position = "none")

ggsave("/research/dataset/AP_Developmental/RateMyProfessor/average_difficulty.jpg",
       height = 3, width = 5, units = "in", dpi = 300, plot = b) 
coded.vars <- setdiff(colnames(df), c("ethnicity", "male", "female", "coded.professor_name", "review_text"))

my_sample_review <- function(my_var = NULL) {
  df2 <- df %>%
    select(c("review_text", my_var)) %>%
    arrange(desc(get(my_var)))
  
  data.frame(Variable = my_var,
             Sample = df2$review_text[1],
             Probability = df2[1,my_var]) %>%
    return()
}

best_examples <-lapply(coded.vars, my_sample_review)
df.best_examples <- do.call(rbind, best_examples)

df.best_examples %>%
  kbl(format = "latex",
      caption = "Prototypical reviews for each automatic code",
      label = "review-samples",
      booktabs = TRUE,
      longtable = TRUE,
      linesep = "",
      align = "l",
      escape = TRUE) %>%
  kable_styling(position = "left",
                latex_options = c("striped", "repeat_header"),
                font_size = 7) %>%
  column_spec(2, width = "40em") %>%
  writeLines('/research/dataset/AP_Developmental/RateMyProfessor/review-samples.tex')

for_vic_top_quality <- df %>%
  filter(quality >= 4 & quality <= 5)

for_vic_top_quality %>%
  group_by(professor_name) %>%
  summarise(ethnicity = head(ethnicity, 1),
            quality = mean(quality)) %>%
  group_by(ethnicity) %>%
  summarise(N = n(),
            quality = mean(quality))

for_vic_mid_quality <- df %>%
  filter(quality >= 2.0 & quality <= 3.0)

for_vic_mid_quality %>%
  group_by(professor_name) %>%
  summarise(ethnicity = head(ethnicity, 1),
            quality = mean(quality)) %>%
  group_by(ethnicity) %>%
  summarise(N = n(),
            quality = mean(quality))

for_vic_bottom_quality <- df %>%
  filter(quality >= 0 & quality <= 1)

for_vic_bottom_quality %>%
  group_by(professor_name) %>%
  summarise(ethnicity = head(ethnicity, 1),
            quality = mean(quality)) %>%
  group_by(ethnicity) %>%
  summarise(N = n(),
            quality = mean(quality))

for_vic_easiest <- df %>%
  filter(difficulty >= 0 & difficulty <= 1)

for_vic_easiest %>%
  group_by(professor_name) %>%
  summarise(ethnicity = head(ethnicity, 1),
            difficulty = mean(difficulty)) %>%
  group_by(ethnicity) %>%
  summarise(N = n(),
            difficulty = mean(difficulty))

for_vic_mid_difficult <- df %>%
  filter(difficulty >= 2.0 & difficulty <= 3.0)

for_vic_mid_difficult %>%
  group_by(professor_name) %>%
  summarise(ethnicity = head(ethnicity, 1),
            difficulty = mean(difficulty)) %>%
  group_by(ethnicity) %>%
  summarise(N = n(),
            difficulty = mean(difficulty))

for_vic_difficult <- df %>%
  filter(difficulty >= 4 & difficulty <= 5)

for_vic_difficult %>%
  group_by(professor_name) %>%
  summarise(ethnicity = head(ethnicity, 1),
            difficulty = mean(difficulty)) %>%
  group_by(ethnicity) %>%
  summarise(N = n(),
            difficulty = mean(difficulty))

saveRDS(for_vic_top_quality, "/research/dataset/AP_Developmental/RateMyProfessor/for_vic_top_quality.rds")
saveRDS(for_vic_mid_quality, "/research/dataset/AP_Developmental/RateMyProfessor/for_vic_mid_quality.rds")
saveRDS(for_vic_bottom_quality, "/research/dataset/AP_Developmental/RateMyProfessor/for_vic_bottom_quality.rds")
saveRDS(for_vic_easiest, "/research/dataset/AP_Developmental/RateMyProfessor/for_vic_easiest.rds")
saveRDS(for_vic_mid_difficult, "/research/dataset/AP_Developmental/RateMyProfessor/for_vic_mid_difficult.rds")
saveRDS(for_vic_difficult, "/research/dataset/AP_Developmental/RateMyProfessor/for_vic_difficult.rds")

######
if (FALSE) {
  vad_words <- lexicon_nrc_vad()

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
  
  num_cores <- detectCores() - 2
  vad_scores <- mclapply(df$review_text, sentence_vad, mc.cores = num_cores)
  df_vad_scores <- do.call(rbind, vad_scores)
  
  search_df <- data.frame(df, df_vad_scores)
  
  #high toxic high valence
  filter(search_df, (toxic > 0.9)) %>%
    arrange(desc(V)) %>%
    head(1)
  
  #high love
  filter(search_df, (love > 0.9)) %>%
    arrange(V) %>%
    head(1)
  
  #high suicidal
  filter(search_df, (suicidal > 0.9)) %>%
    arrange(desc(V)) %>%
    head(1)
  
  summary(lm(search_df$positive_sentiment ~ search_df$A))
}


