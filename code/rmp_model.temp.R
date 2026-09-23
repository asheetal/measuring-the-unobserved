#!/usr/bin/Rscript

Sys.setenv("CUDA_VISIBLE_DEVICES"=1)
suppressPackageStartupMessages({
  library(tidyverse)
  library(janitor)
  library(data.table)
  library(mltools)
  library(word2vec)
  library(filenamer)
  library(keras)
  library(reticulate)
})

MAX_WORDS <- 512L
MAXLEN <- 512L
DIM_SIZE <- 300L
options(filenamer.timestamp=1)
basedir <- "/research/dataset/AP_Developmental/RateMyProfessor/"

FLAGS <- flags(
  flag_numeric("R_FLAGS_GRU_UNITS", 256),
  flag_numeric("R_FLAGS_MY_UNITS1", 1000),
  flag_numeric("R_FLAGS_MY_UNITS2", 500),
  flag_numeric("R_FLAGS_MY_UNITS3", 200),
  flag_numeric("R_FLAGS_MY_UNITS4", 200),
  flag_numeric("R_FLAGS_MY_UNITS5", 100),
  flag_numeric("R_FLAGS_MY_UNITS6", 50),
  flag_numeric("R_FLAGS_MY_UNITS7", 10),
  flag_numeric("R_FLAGS_DROPOUT", 0.1),
  flag_numeric("R_FLAGS_MY_BATCHSIZE", 256),
  flag_numeric("R_FLAGS_REGULARIZER", 0.01),
  flag_numeric("R_FLAGS_MY_LR_MAN", 4),
  flag_numeric("R_FLAGS_MY_LR_EXP", -5)
)
my_lr <- as.numeric(FLAGS$R_FLAGS_MY_LR_MAN) * 10^(FLAGS$R_FLAGS_MY_LR_EXP)

rmp <- readRDS("/research/dataset/AP_Developmental/RateMyProfessor/rmp.cleaned.coded.rds")

set.seed(101)
df.train <- rmp[[1]] %>%
  select(-c("coded.professor_name")) %>%
  mutate(ethnicity = as.factor(ethnicity)) %>%
  slice(sample(1:n()))

CODED_CONSTRUCTS <- colnames(df.train) %>%
  setdiff(c("review_text", "ethnicity"))

last_col <- ncol(df.train)
x_train.constructs <- df.train[,CODED_CONSTRUCTS] %>%
  as.matrix()

# extract the output
y_train = df.train %>%
  select(ethnicity) %>%
  as.data.table() %>%
  one_hot() %>%
  as.matrix()

sklearn <- import("sklearn.utils")
np <- import("numpy", convert=FALSE)
compute_class_weights <- sklearn$compute_class_weight

focal_loss_class_weights <- compute_class_weights(class_weight='balanced', 
                                                  classes=np$unique(df.train$ethnicity), 
                                                  y=np$array(df.train$ethnicity))

class_weights <- as.list(focal_loss_class_weights)
names(class_weights) <- 0:3

weight_table <-  data.frame(table(df.train$ethnicity))
weight_table$weights = min(weight_table$Freq) / weight_table$Freq
sample_weight <- weight_table[df.train$ethnicity, "weights"]


# Number of instances for each class
class_counts <- weight_table$Freq
# Calculate frequencies
total_instances <- sum(class_counts)
class_frequencies <- class_counts / total_instances
# Calculate inverse frequencies
inverse_frequencies <- 1 / class_frequencies
# Normalize weights
WEIGHTS <- inverse_frequencies / sum(inverse_frequencies)

source("/research/SRC/AP_Developmental/loss_functions.R")
#my_loss <- tensorflow::tf$keras$losses$CategoricalFocalCrossentropy(alpha = focal_loss_class_weights)

my_get_glove_vectors <- function(path = NULL) {
  g.model <- read.word2vec(path)
  g.embedding <- as.matrix(g.model)
  g.vectors <- data.frame(rownames(g.embedding), g.embedding)
  colnames(g.vectors) <- c('word', paste('dim',1:300,sep = '_'))
  return(g.vectors)
}

vectors.asian <- my_get_glove_vectors("/research/dataset/AP_Developmental/RateMyProfessor/word2vec_black.42B.model")
vectors.black <- my_get_glove_vectors("/research/dataset/AP_Developmental/RateMyProfessor/word2vec_black.42B.model")
vectors.hispanic <- my_get_glove_vectors("/research/dataset/AP_Developmental/RateMyProfessor/word2vec_hispanic.42B.model")
vectors.white <- my_get_glove_vectors("/research/dataset/AP_Developmental/RateMyProfessor/word2vec_white.42B.model")

# tokenize the input data and then fit the created object
word_seqs = text_tokenizer(num_words = MAX_WORDS) %>%
  fit_text_tokenizer(df.train$review_text)

# apply tokenizer to the text and get indices instead of words
# later pad the sequence
x_train.text <- texts_to_sequences(word_seqs, df.train$review_text) %>%
  pad_sequences(maxlen = MAXLEN)

# unlist word indices
word_indices = unlist(word_seqs$word_index)

# then place them into data.frame 
dic = data.frame(word = names(word_indices), key = word_indices, stringsAsFactors = FALSE) %>%
  arrange(key) %>% .[1:MAX_WORDS,]

# join the words with GloVe vectors and
# if word does not exist in GloVe, then fill NA's with 0
word_embeds.asian = dic  %>% 
  left_join(vectors.asian, by = c("word" = "word")) %>%
  .[,3:302] %>% 
  replace(., is.na(.), 0) %>% 
  as.matrix()

word_embeds.black = dic  %>% 
  left_join(vectors.black, by = c("word" = "word")) %>%
  .[,3:302] %>% 
  replace(., is.na(.), 0) %>% 
  as.matrix()

word_embeds.hispanic = dic  %>% 
  left_join(vectors.hispanic, by = c("word" = "word")) %>%
  .[,3:302] %>% 
  replace(., is.na(.), 0) %>% 
  as.matrix()

word_embeds.white = dic  %>% 
  left_join(vectors.white, by = c("word" = "word")) %>%
  .[,3:302] %>% 
  replace(., is.na(.), 0) %>% 
  as.matrix()

my_layers <- function(m = NULL) {
  m2 <- m %>%
    layer_spatial_dropout_1d(rate = FLAGS$R_FLAGS_DROPOUT) %>%
    bidirectional(layer_gru(units = FLAGS$R_FLAGS_GRU_UNITS, 
                            kernel_regularizer = regularizer_l1(FLAGS$R_FLAGS_REGULARIZER),
                            #recurrent_dropout=FLAGS$R_FLAGS_DROPOUT,
                            dropout=FLAGS$R_FLAGS_DROPOUT,
                            return_sequences = TRUE))
  m.max <- m2 %>% layer_global_max_pooling_1d()
  m.ave <- m2 %>% layer_global_average_pooling_1d()
  m.out <- layer_concatenate(list(m.ave, m.max))
  return(m.out)
}


# Use Keras Functional API 
input.text <- layer_input(shape = list(MAXLEN), name = "input.text")

model.asian = input.text %>%
  layer_embedding(input_dim = MAX_WORDS, output_dim = DIM_SIZE, input_length = MAXLEN, 
                  # put weights into list and do not allow training
                  weights = list(word_embeds.asian), trainable = FALSE) %>%
  my_layers()

model.black = input.text %>%
  layer_embedding(input_dim = MAX_WORDS, output_dim = DIM_SIZE, input_length = MAXLEN, 
                  # put weights into list and do not allow training
                  weights = list(word_embeds.black), trainable = FALSE) %>%
  my_layers()

model.hispanic <- input.text %>%
  layer_embedding(input_dim = MAX_WORDS, output_dim = DIM_SIZE, input_length = MAXLEN, 
                  # put weights into list and do not allow training
                  weights = list(word_embeds.hispanic), trainable = FALSE) %>%
  my_layers()


model.white = input.text %>%
  layer_embedding(input_dim = MAX_WORDS, output_dim = DIM_SIZE, input_length = MAXLEN, 
                  # put weights into list and do not allow training
                  weights = list(word_embeds.white), trainable = FALSE) %>%
  
  my_layers()

input.constructs <- layer_input(shape = list(length(CODED_CONSTRUCTS)), name = "input.constructs")

my_keras_layer <- function(object = NULL, n_nodes = NULL) {
  out_object <- object %>%
    layer_dense(units = n_nodes,
                #kernel_regularizer = regularizer_l1(FLAGS$R_FLAGS_REGULARIZER),
                kernel_initializer = "he_uniform") %>%
    layer_activation("relu") %>%
    layer_gaussian_dropout(rate = FLAGS$R_FLAGS_DROPOUT) %>%
    layer_batch_normalization()
  return(out_object)
}

output <- layer_concatenate(list(model.asian, model.black, model.hispanic, model.white, input.constructs)) %>%
  my_keras_layer(FLAGS$R_FLAGS_MY_UNITS1) %>%
  my_keras_layer(FLAGS$R_FLAGS_MY_UNITS2) %>%
  my_keras_layer(FLAGS$R_FLAGS_MY_UNITS3) %>%
  my_keras_layer(FLAGS$R_FLAGS_MY_UNITS4) %>%
  my_keras_layer(FLAGS$R_FLAGS_MY_UNITS5) %>%
  my_keras_layer(FLAGS$R_FLAGS_MY_UNITS6) %>%
  my_keras_layer(FLAGS$R_FLAGS_MY_UNITS7) %>%
  layer_dense(units = ncol(y_train),
              kernel_regularizer = regularizer_l1(FLAGS$R_FLAGS_REGULARIZER),
              kernel_initializer = "he_uniform") %>%
  layer_activation("softmax", name = "ethnicity")

model <- keras_model(inputs = list(input.text, input.constructs), outputs = output)

model %>% compile(
  run_eagerly = TRUE,
  optimizer = optimizer_adam(learning_rate=my_lr),
  loss = "categorical_crossentropy",
  #loss = "categorical_focal_crossentropy",
  #loss = my_categorical_hinge_loss,
  #loss = my_categorical_focal_loss,
  #loss = my_dice(),
  #loss = weighted_dice_loss(smooth = 1.0, class_weights = WEIGHTS),
  weighted_metrics = c("accuracy", "AUC", F1)
)

model_file <- filename("model_output",
                       path=basedir,
                       tag=NULL,
                       ext="hdf5",
                       subdir=FALSE) %>%
  as.character() %>%
  print()

callbacks <- list(callback_early_stopping(monitor="val_loss", 
                                          patience = 40),
                  callback_reduce_lr_on_plateau(monitor="val_loss", 
                                                patience=10, 
                                                factor = 0.5),
                  callback_model_checkpoint(filepath = model_file,
                                            monitor = "val_loss",
                                            save_best_only = TRUE,
                                            save_freq = "epoch",
                                            verbose = 0,
                                            mode = "min"))

history <- model %>% 
  fit(x = list(x_train.text, x_train.constructs), 
      y = y_train,
      sample_weight = sample_weight,
      #class_weight = class_weights,
      epochs = 200L,
      batch_size = FLAGS$R_FLAGS_MY_BATCHSIZE,
      validation_split = 0.2,
      verbose = 1,
      callbacks = callbacks
  )

options(filenamer.timestamp=1)
training_image <- filename("training",
                           path=basedir,
                           tag=NULL,
                           ext="png",
                           subdir=FALSE) %>%
  as.character() %>%
  print()
png(filename=training_image, 
    height = 8, width = 12, res = 300, units = "in")
print(plot(history, metrics = c("loss", "accuracy", "auc", "F1")))
dev.off()
