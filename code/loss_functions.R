F1 <- custom_metric("F1", function(y_true, y_pred) {
  # Convert predictions to one-hot encoded format if they are not already
  y_pred <- k_round(y_pred)
  
  # Calculate metrics for each class
  F1_scores <- k_constant(0)
  for (class_index in 1:ncol(y_true)) {
    y_true_class <- y_true[, class_index]
    y_pred_class <- y_pred[, class_index]
    
    y_correct <- y_true_class * y_pred_class
    sum_true <- k_sum(y_true_class)
    sum_pred <- k_sum(y_pred_class)
    sum_correct <- k_sum(y_correct)
    epsilon <- k_epsilon()
    
    precision <- sum_correct / (sum_pred + epsilon)
    recall <- sum_correct / (sum_true + epsilon)
    
    # Compute F1 score for the current class
    F1_class <- 2 * (precision * recall) / (precision + recall + epsilon)
    
    # Accumulate F1 scores across all classes
    F1_scores <- F1_scores + F1_class
  }
  
  # Calculate macro F1 score by averaging across all classes
  macro_F1 <- F1_scores / ncol(y_true)
  return(macro_F1)
})