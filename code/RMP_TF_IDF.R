library(tidyverse)
library(tidytext)
library(forcats)
library(lexicon)

#psych_terms <- lexicon::hash_sentiment_nrc
psych_terms <- lexicon_nrc_vad()

rmp.train <- readRDS("/research/dataset/AP_Developmental/RateMyProfessor/rmp.cleaned.coded.rds")

df.train.groups <- rmp.train[["train"]] %>%
  #filter(ethnicity %in% c("Asian", "Black", "Hispanic", "White")) %>%
  filter(nchar(review_text) > 10) %>%
  select(c("review_text", "ethnicity")) %>%
  rename(text = review_text,
         book = ethnicity) %>%
  mutate(book = factor(book)) %>%
  as_tibble()

# Run the function
results <- find_common_words(df.split.groups)

# View the common words
print(results$common_words$word)

book_words <- df.train.groups %>%
  unnest_tokens(word, text) %>%
  count(book, word, sort = TRUE) %>%
  filter(word %in% psych_terms$Word)


total_words <- book_words %>% group_by(book) %>% summarize(total = sum(n))
book_words <- left_join(book_words, total_words)
book_words

ggplot(book_words, aes(n/total, fill = book)) +
  geom_histogram(show.legend = FALSE) +
  scale_x_continuous(limits = c(NA, 0.0009)) +
  facet_wrap(vars(book), ncol = 2, scales = "free_y") +
  theme_minimal()

freq_by_rank <- book_words %>% 
  group_by(book) %>% 
  mutate(rank = row_number(), 
         term_frequency = n/total) %>%
  ungroup()
freq_by_rank

freq_by_rank %>% 
  ggplot(aes(rank, term_frequency, color = book)) + 
  geom_line(linewidth = 1.1, alpha = 0.8, show.legend = FALSE) + 
  scale_x_log10() +
  scale_y_log10()

rank_subset <- freq_by_rank %>% 
  filter(rank < 500,
         rank > 10)

lm(log10(term_frequency) ~ log10(rank), data = rank_subset)

freq_by_rank %>% 
  ggplot(aes(rank, term_frequency, color = book)) + 
  geom_abline(intercept = -0.62, slope = -1.1, 
              color = "gray50", linetype = 2) +
  geom_line(linewidth = 1.1, alpha = 0.8, show.legend = FALSE) + 
  scale_x_log10() +
  scale_y_log10()

book_tf_idf <- book_words %>%
  bind_tf_idf(word, book, n)

book_tf_idf

book_tf_idf %>%
  select(-total) %>%
  arrange(desc(tf_idf))

book_tf_idf %>%
  group_by(book) %>%
  slice_max(tf_idf, n = 15) %>%
  ungroup() %>%
  ggplot(aes(tf_idf, fct_reorder(word, tf_idf), fill = book)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~book, ncol = 2, scales = "free") +
  labs(x = "tf-idf", y = NULL)

ggsave("/research/dataset/AP_Developmental/RateMyProfessor/tf-idf.jpg",
       width = 8.5, height = 6, dpi = 300, units = "in")
