suppressPackageStartupMessages({
  library(tidyverse)
  library(probably)
  library(caret)
  library(betacal)
  library(mltools)
  library(data.table)
  library(scales)
  library(reticulate)
  library(rfUtilities)
})

TYPE <- Sys.getenv("TYPE") %>% as.integer()
print(paste("Working on TYPE ", TYPE))

for_cfm <- readRDS(case_when(TYPE==1 ~ "/research/dataset/AP_Developmental/RateMyProfessor/for_cfm_1.rds",
                             TYPE==2 ~ "/research/dataset/AP_Developmental/RateMyProfessor/for_cfm_2.rds",
                             TRUE ~ "/research/dataset/AP_Developmental/RateMyProfessor/for_cfm.rds"))

weight_table <-  data.frame(table(for_cfm[["y.train"]]))
weight_table$weights = min(weight_table$Freq) / weight_table$Freq
sample_weight <- weight_table[for_cfm[["y.test"]], "weights"]

y.test.hat.probs <- for_cfm[["y_hat.test_probs"]]

y.test.hat <- colnames(y.test.hat.probs)[apply(y.test.hat.probs,1,which.max)] %>%
  factor(levels = levels(for_cfm[["y.train"]]))

sklearn.metrics <- import("sklearn.metrics")

cfm <- sklearn.metrics$confusion_matrix(for_cfm[["y.test"]], y.test.hat, sample_weight = sample_weight) %>%
  round()
rownames(cfm) <- levels(for_cfm[["y.train"]])
colnames(cfm) <- levels(for_cfm[["y.train"]])
names(dimnames(cfm)) <- c("Reference", "Prediction")

cfmw <- confusionMatrix(as.table(cfm))

{
  sink(paste0("/research/dataset/AP_Developmental/RateMyProfessor/cfm_weighted_", TYPE, ".txt"))
  
  print(confusionMatrix(as.table(cfm)))
  cat("\n")
  paste0("\nWeighted accuracy score is ", round(sklearn.metrics$accuracy_score(for_cfm[["y.test"]], y.test.hat, sample_weight = sample_weight), 3)) %>%
    cat()
  
  paste0("\nWeighted F1 score is ", round(sklearn.metrics$f1_score(for_cfm[["y.test"]], y.test.hat, average = "weighted", sample_weight = sample_weight), 3)) %>%
    cat()
  
  paste0("\nWeighted Cohen's Kappa is ", round(sklearn.metrics$cohen_kappa_score(for_cfm[["y.test"]], y.test.hat, sample_weight = sample_weight), 3)) %>%
    cat()
  
  paste0("\nWeighted AUC is ", round(sklearn.metrics$roc_auc_score(for_cfm[["y.test"]], y.test.hat.probs, multi_class = "ovr",
                                                                   average = "weighted", sample_weight = sample_weight), 3)) %>%
    cat()
  
  
  sink()
}


ggplotConfusionMatrix <- function(m, my_auc = NULL, my_f1 = NULL){
  mytitle <- paste("Accuracy", percent_format()(m$overall[1]),
                   "F1", percent_format()(my_f1),
                   "Kappa", percent_format()(m$overall[2]),
                   "AUC", percent_format()(my_auc))
  p <-
    ggplot(data = as.data.frame(m$table) ,
           aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = log(Freq)), colour = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
    theme(legend.position = "none", axis.title = element_text(size = 10), plot.title = element_text(size = 10)) +
    ggtitle(mytitle)
  return(p)
}

my_auc <- sklearn.metrics$roc_auc_score(for_cfm[["y.test"]], y.test.hat.probs, multi_class = "ovr",
                                        average = "weighted", sample_weight = sample_weight)

my_f1 <- sklearn.metrics$f1_score(for_cfm[["y.test"]], y.test.hat, average = "weighted", sample_weight = sample_weight)

confusionMatrix(as.table(cfm)) %>%
  ggplotConfusionMatrix(my_auc, my_f1)

ggsave(paste0("/research/dataset/AP_Developmental/RateMyProfessor/cfmw_", TYPE, ".jpg"), 
       width = 5.5, height = 2, units =  "in", dpi=300)
