library(reticulate)
library(tidyverse)
library(DT)
library(readxl)
library(janitor)
library(doParallel)
library(foreach)
library(future)
options(future.globals.onReference = "error")
NCPU <- 31L
Sys.setenv(TOKENIZERS_PARALLELISM="true")
#https://huggingface.co/SamLowe/roberta-base-go_emotions
add_emotions <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("SamLowe/roberta-base-go_emotions", use_cache=T)
    model <- autoModelClass$from_pretrained("SamLowe/roberta-base-go_emotions", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(admiration = .[1],
                                        amusement = .[2],
                                        anger = .[3],
                                        annoyance = .[4],
                                        approval = .[5],
                                        caring = .[6],
                                        confusion = .[7],
                                        curiosity = .[8],
                                        desire = .[9],
                                        disappointment = .[10],
                                        disapproval = .[11],
                                        disgust = .[12],
                                        embarassment = .[13],
                                        excitement = .[14],
                                        fear = .[15],
                                        gratitude = .[16],
                                        grief = .[17],
                                        joy = .[18],
                                        love = .[19],
                                        nervousness = .[20],
                                        optimism = .[21],
                                        pride = .[22],
                                        realization = .[23],
                                        relief = .[24],
                                        remorse = .[25],
                                        sadness = .[26],
                                        surprise = .[27],
                                        neutral = .[28]))
  
  return(table)
}

#https://huggingface.co/textdetox/xlmr-large-toxicity-classifier
add_toxicity <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("textdetox/xlmr-large-toxicity-classifier", use_cache=T)
    model <- autoModelClass$from_pretrained("textdetox/xlmr-large-toxicity-classifier", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  
  table <- map_dfr(pred_table, ~ tibble(not_toxic = .[1],
                                        toxic = .[2]))
  return(table)
}

#https://huggingface.co/cardiffnlp/twitter-roberta-base-irony
add_irony <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("cardiffnlp/twitter-roberta-base-irony", use_cache=T)
    model <- autoModelClass$from_pretrained("cardiffnlp/twitter-roberta-base-irony", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation = TRUE, return_tensors='pt', max_length = 512L) # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  
  table <- map_dfr(pred_table, ~ tibble(irony = .[1],
                                        non_irony = .[2]))
  return(table)
}

#Abusive
#https://huggingface.co/Hate-speech-CNERG/english-abusive-MuRIL
add_abusive <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("Hate-speech-CNERG/english-abusive-MuRIL", use_cache=T)
    model <- autoModelClass$from_pretrained("Hate-speech-CNERG/english-abusive-MuRIL", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation = TRUE, return_tensors='pt', max_length = 512L) # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  table <- map_dfr(pred_table, ~ tibble(non_abusive = .[1],
                                        abusive = .[2]))
  return(table)
}

#suicidality
#https://huggingface.co/sentinet/suicidality
add_suicidal <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("sentinet/suicidality", use_cache=T)
    model <- autoModelClass$from_pretrained("sentinet/suicidality", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(non_suicidal = .[1],
                                        suicidal = .[2]))
  return(table)
}

#Big5 Personality
#https://huggingface.co/KevSun/Personality_LM
add_personality <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("KevSun/Personality_LM", use_cache=T)
    model <- autoModelClass$from_pretrained("KevSun/Personality_LM", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation = TRUE, return_tensors='pt', max_length = 512L) # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(agreeableness = .[1],
                                        openness = .[2],
                                        conscientiousness = .[3],
                                        extraversion = .[4],
                                        neuroticism = .[5]))
  return(table)
}

#https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
add_sentiment <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", use_cache=T)
    model <- autoModelClass$from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation = TRUE, return_tensors='pt', max_length = 512L) # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(negative_sentiment = .[1],
                                        neutral_sentiment = .[2],
                                        positive_sentiment = .[3]))
  return(table)
}


#https://huggingface.co/parsawar/profanity_model_3.1
add_profanity <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("parsawar/profanity_model_3.1", use_cache=T)
    model <- autoModelClass$from_pretrained("parsawar/profanity_model_3.1", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation = TRUE, return_tensors='pt', max_length = 512L) # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(non_offensive = .[1],
                                        offensive = .[2]))
  return(table)
}


#https://huggingface.co/vtiyyal1/empathy_model"
add_empathy <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("vtiyyal1/empathy_model", use_cache=T)
    model <- autoModelClass$from_pretrained("vtiyyal1/empathy_model")
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$sigmoid(outputs$logits)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(empathy = .[1]))
  return(table)
}

#https://huggingface.co/amedvedev/bert-tiny-cognitive-bias
add_cognitive_bias <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("amedvedev/bert-tiny-cognitive-bias", use_cache=T)
    model <- autoModelClass$from_pretrained("amedvedev/bert-tiny-cognitive-bias", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation = TRUE, return_tensors='pt', max_length = 512L) # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(no_distortion = .[1],
                                        personalization = .[2],
                                        emotional_reasoning = .[3],
                                        overgeneralizing = .[4],
                                        labeling = .[5],
                                        should_statements = .[6],
                                        catastrophizing = .[7],
                                        reward_fallacy = .[8]))
  return(table)
}

#https://huggingface.co/knkarthick/Action_Decisions
add_action_decision <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("knkarthick/Action_Decisions", use_cache=T)
    model <- autoModelClass$from_pretrained("knkarthick/Action_Decisions")
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(not_action_decision = .[1],
                                        action_decision = .[2]))
  return(table)
}

#https://huggingface.co/Sami92/XLM-R-Large-Sensationalism-Classifier
add_sensationalism <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("Sami92/XLM-R-Large-Sensationalism-Classifier", use_cache=T)
    model <- autoModelClass$from_pretrained("Sami92/XLM-R-Large-Sensationalism-Classifier", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(not_sensational = .[1],
                                        sensational = .[2]))
  return(table)
}

#https://huggingface.co/rafalposwiata/deproberta-large-depression
add_depression <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("rafalposwiata/deproberta-large-depression", use_cache=T)
    model <- autoModelClass$from_pretrained("rafalposwiata/deproberta-large-depression", use_cache=T)
    inputs <- tokenizer(df[i, ]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  
  table <- map_dfr(pred_table, ~ tibble(not_depressed = .[1],
                                        moderately_depressed = .[2],
                                        severely_depressed = .[3]))
  return(table)
}

#https://huggingface.co/michellejieli/NSFW_text_classifier
add_nsfw <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("michellejieli/NSFW_text_classifier", use_cache=T)
    model <- autoModelClass$from_pretrained("michellejieli/NSFW_text_classifier")
    inputs <- tokenizer(df[i, ]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(nsfw = .[1],
                                        non_nsfw = .[2]))
  return(table)
}

#https://huggingface.co/jackhhao/jailbreak-classifier
add_jailbreak <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("jackhhao/jailbreak-classifier", use_cache=T)
    model <- autoModelClass$from_pretrained("jackhhao/jailbreak-classifier", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(jailbreak_benign = .[1],
                                        jailbreak = .[2]))
  return(table)
}

#https://huggingface.co/leondz/refutation_detector_distilbert
add_refutation <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("leondz/refutation_detector_distilbert", use_cache=T)
    model <- autoModelClass$from_pretrained("leondz/refutation_detector_distilbert")
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(refutation = .[1],
                                        non_refutation = .[2]))
  return(table)
}

#https://huggingface.co/helinivan/english-sarcasm-detector
add_sarcasm <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("helinivan/english-sarcasm-detector", use_cache=T)
    model <- autoModelClass$from_pretrained("helinivan/english-sarcasm-detector", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(non_sarcasm = .[1],
                                        sarcasm = .[2]))
  return(table)
}

#https://huggingface.co/maximuspowers/bias-type-classifier
add_biases <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("maximuspowers/bias-type-classifier", use_cache=T)
    model <- autoModelClass$from_pretrained("maximuspowers/bias-type-classifier", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(bias_racial = .[1],
                                        bias_religious = .[2],
                                        bias_gender = .[3],
                                        bias_age = .[4],
                                        bias_nationality = .[5],
                                        bias_sexuality = .[6],
                                        bias_socioeconomic = .[7],
                                        bias_educational = .[8],
                                        bias_disability = .[9],
                                        bias_political = .[10],
                                        bias_physical = .[11]))
  return(table)
}

#https://huggingface.co/dejanseo/good-vibes
add_vibes <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("dejanseo/good-vibes", use_cache=T)
    model <- autoModelClass$from_pretrained("dejanseo/good-vibes")
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(good_vibes = .[1],
                                        no_vibes = .[2],
                                        bad_vibes = .[3]))
  return(table)
}

#https://huggingface.co/civility-lab/roberta-base-namecalling
add_namecalling <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("civility-lab/roberta-base-namecalling", use_cache=T)
    model <- autoModelClass$from_pretrained("civility-lab/roberta-base-namecalling", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(non_namecalling = .[1],
                                        namecalling = .[2]))
  return(table)
}

#https://huggingface.co/holistic-ai/rejection_detection
add_rejection <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("holistic-ai/rejection_detection", use_cache=T)
    model <- autoModelClass$from_pretrained("holistic-ai/rejection_detection", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(non_rejection = .[1],
                                        rejection = .[2]))
  return(table)
}

#https://huggingface.co/Falconsai/fear_mongering_detection
add_fear_mongering <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("Falconsai/fear_mongering_detection", use_cache=T)
    model <- autoModelClass$from_pretrained("Falconsai/fear_mongering_detection")
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(non_fear_mongering = .[1],
                                        fear_mongering = .[2]))
  return(table)
}

#https://huggingface.co/chrlukas/flattery_prediction_text
add_flattery <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("chrlukas/flattery_prediction_text", use_cache=T)
    model <- autoModelClass$from_pretrained("chrlukas/flattery_prediction_text", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$sigmoid(outputs$logits)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(flattery = .[1]))
  return(table)
}

#https://huggingface.co/tee-oh-double-dee/social-orientation
add_social_orientation <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("tee-oh-double-dee/social-orientation", use_cache=T)
    model <- autoModelClass$from_pretrained("tee-oh-double-dee/social-orientation")
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(social_cold = .[1],
                                        social_arrogant_calculating = .[2],
                                        social_aloof_introverted = .[3],
                                        social_assured_dominant = .[4],
                                        social_unassuming_ingenuous =.[5],
                                        social_unassured_submissive=.[6],
                                        social_warm_agreeable = .[7],
                                        social_gregarious_extraverted = .[8],
                                        social_not_available = .[9]))
  return(table)
}

#https://huggingface.co/Reggie/muppet-roberta-base-joke_detector
add_joke <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("Reggie/muppet-roberta-base-joke_detector", use_cache=T)
    model <- autoModelClass$from_pretrained("Reggie/muppet-roberta-base-joke_detector", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation = TRUE, return_tensors='pt', max_length = 512L) # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(not_joke = .[1],
                                        joke = .[2]))
  return(table)
}

#https://huggingface.co/CommunicationStyle/Communication_Style
add_communication <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("CommunicationStyle/Communication_Style", use_cache=T)
    model <- autoModelClass$from_pretrained("CommunicationStyle/Communication_Style", use_cache=T)
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation = TRUE, return_tensors='pt', max_length = 512L) # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(comm_communion = .[1],
                                        comm_agency = .[2],
                                        comm_none = .[3]))
  return(table)
}

#https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457
add_gibberish <- function(df = NULL) {
  cluster <- makeCluster(NCPU) 
  registerDoParallel(cluster)
  pred_table <- foreach (i = 1:nrow(df)) %dopar% {
    transformer <- reticulate::import("transformers")
    torch <- reticulate::import("torch")
    autotoken <- transformer$AutoTokenizer
    autoModelClass <- transformer$AutoModelForSequenceClassification
    tokenizer <- autotoken$from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457", use_cache=T)
    model <- autoModelClass$from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457")
    inputs <- tokenizer(df[i,]$text, padding=TRUE, truncation=TRUE, return_tensors='pt') # pt stands for pytorch
    outputs <- model(inputs$input_ids, attention_mask=inputs$attention_mask)
    predictions <- torch$nn$functional$softmax(outputs$logits, dim=1L)
    rm(transformer, torch, autotoken, autoModelClass, tokenizer, model, inputs, outputs)
    gc()
    unlist(predictions$tolist())
  }
  stopCluster(cluster)
  table <- map_dfr(pred_table, ~ tibble(gibberish_clean = .[1],
                                        gibberish_mild = .[2],
                                        gibberish_noise = .[3],
                                        gibberish_word_salad = .[4]))
  return(table)
}

