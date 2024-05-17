# D208 - Predictive Modeling

# rm(list = ls())

# Define the CRAN repository URL
cran_repo <- "https://cran.r-project.org"

# Install and load the required packages/libraries
install.packages("dplyr", repos = cran_repo, quiet = TRUE)
library(dplyr)

library(stats)

install.packages("tidyverse", repos = cran_repo, quiet = TRUE)
library(purrr)

install.packages("ggplot2", repos = cran_repo, quiet = TRUE)
library(ggplot2)

install.packages("gridExtra", repos = cran_repo, quiet = TRUE)
library(gridExtra)
library(grid)

install.packages("glmnet", repos = cran_repo, quiet = TRUE)
install.packages("glmnet", repos = cran_repo, quiet = TRUE)  # This line seems redundant

library(glmnet)

install.packages("caret", repos = cran_repo, quiet = TRUE)
library(caret)

install.packages("car")
library(car)

library(e1071) # May be needed for dependency reasons
install.packages("randomForest")
library(randomForest)

install.packages("summarytools")

library(summarytools)
install.packages("fastDummies")
library(fastDummies)

# Define the correct file path to medical_raw_data.csv
#file_path <- "/Users/john/Downloads/medical_clean.csv"
file_path <- "C:/Users/John/Desktop/medical_clean.csv"

# Read the CSV file into a variable (e.g., 'medical_data')
medical_data <- read.csv(file_path)


# Filter 'medical_data' to include only the specified states
medical_data <- medical_data %>%
  filter(State %in% c('DC', 'VA', 'DE', 'MD'))
medical_data_csv <- medical_data
#Convert Categorical variables into numerical data points

categorical_variables <- c(
  "City", "State", "County", "Zip", "Area", "TimeZone", "Job", "Marital",
  "Gender", "ReAdmis", "Soft_drink", "Initial_admin", "HighBlood", "Stroke",
  "Complication_risk", "Overweight", "Arthritis", "Diabetes", "Hyperlipidemia",
  "BackPain", "Anxiety", "Allergic_rhinitis", "Reflux_esophagitis", "Asthma",
  "Services", "Item1", "Item2", "Item3", "Item4", "Item5", "Item6", "Item7", "Item8"
)

continuous_variables <- c(
  "Population", "Children", "Age", "Income", "VitD_levels", "Doc_visits",
  "Full_meals_eaten", "vitD_supp", "Initial_days", "TotalCharge", "Additional_charges"
)

# Summary statistics for categorical variables
summary_categorical <- summary(medical_data[categorical_variables])

# Summary statistics for continuous variables
summary_continuous <- summary(medical_data[continuous_variables])

#Continous - Continuous Variable Check using correlation matrix
continuous_data <- medical_data[, continuous_variables, drop = FALSE]

# Compute pairwise correlations between continuous variables
correlation_matrix <- cor(continuous_data)

#Continuous - Categorical Variable Check using Cross-tabulation and Fisher Test
#Cross-tabulation - examine the relationship between pairs of categorical variables
for (cat_var in categorical_variables) {
  # Convert the variable to factor if it's not already
  if (!is.factor(medical_data[[cat_var]])) {
    medical_data[[cat_var]] <- as.factor(medical_data[[cat_var]])
  }
  
  # Plot the distribution of the categorical variable
  ggplot(medical_data, aes(x = !!sym(cat_var))) +
    geom_bar() +
    labs(title = cat_var)
}

# Cross-tabulation - examine the relationship between pairs of categorical variables
# Create an empty list to store significant results
# Initialize lists to store significant and non-significant results
significant_results <- list()
non_significant_results <- list()

for (i in 1:(length(categorical_variables) - 1)) {
  for (j in (i+1):length(categorical_variables)) {
    cat_var1 <- categorical_variables[i]
    cat_var2 <- categorical_variables[j]
    
    # Check if both variables have at least two levels
    if (length(unique(medical_data[[cat_var1]])) >= 2 && length(unique(medical_data[[cat_var2]])) >= 2) {
      
      # Create contingency table
      cross_tab <- table(medical_data[[cat_var1]], medical_data[[cat_var2]])
      
      # Fisher's exact test with Monte Carlo simulation
      fisher_test <- fisher.test(cross_tab, simulate.p.value = TRUE, B = 3000)  # Adjust B as needed
      
      # Check if the simulated p-value is less than or equal to 0.05
      if (fisher_test$p.value <= 0.05) {
        # Store significant results
        significant_results[[paste(cat_var1, cat_var2, sep = "_vs_")]] <- list(
          contingency_table = cross_tab,
          fisher_exact_test = fisher_test
        )
      } else {
        # Store non-significant results
        non_significant_results[[paste(cat_var1, cat_var2, sep = "_vs_")]] <- list(
          contingency_table = cross_tab,
          fisher_exact_test = fisher_test
        )
      }
    }
  }
}


#Store the variables that pass the checks. Categorical Data and Continuous Data
# Initialize a vector to store p-values
p_values <- c()

# Loop through significant results and extract p-values
for (pair in names(significant_results)) {
  p_value <- significant_results[[pair]]$fisher_exact_test$p.value
  p_values <- c(p_values, p_value)
}

# Sort p-values in descending order
sorted_p_values <- sort(p_values, decreasing = TRUE)

# Identify the five variables with the highest p-values
top_ten <- head(sorted_p_values, 10)

# Initialize an empty list to store variable names and their p-values
top_ten_info <- list()

# Get the variable names and p-values corresponding to the top 5 p-values
for (p_value in top_ten) {
  for (pair in names(significant_results)) {
    if (significant_results[[pair]]$fisher_exact_test$p.value == p_value) {
      top_ten_info[[pair]] <- list(
        variable_pair = pair,
        p_value = p_value
      )
      break
    }
  }
}

#For nonsignificant
# Initialize a vector to store p-values
p_values_non_significant <- c()

# Loop through non-significant results and extract p-values
for (pair in names(non_significant_results)) {
  p_value <- non_significant_results[[pair]]$fisher_exact_test$p.value
  p_values_non_significant <- c(p_values_non_significant, p_value)
}

# Sort p-values in descending order
sorted_p_values_non_significant <- sort(p_values_non_significant, decreasing = TRUE)

# Identify the ten variables with the highest p-values
top_ten_non_significant <- head(sorted_p_values_non_significant, 10)

# Initialize an empty list to store variable names and their p-values
top_ten_info_non_significant <- list()

# Get the variable names and p-values corresponding to the top ten p-values
for (p_value in top_ten_non_significant) {
  for (pair in names(non_significant_results)) {
    if (non_significant_results[[pair]]$fisher_exact_test$p.value == p_value) {
      top_ten_info_non_significant[[pair]] <- list(
        variable_pair = pair,
        p_value = p_value
      )
      break
    }
  }
}

#Any variable pairings in the significant list should not be included in the research variable list
continuous_variables <- c("Population", "Children", "Age", "VitD_levels", "Doc_visits", "Income","TotalCharge","Initial_days","Additional_charges")

# Null hypothesis - Predictor variables are independent of each other
#Categorical Variables - highest p-values. 10 variables from top_five_info. 
categorical_variables <- c("Reflux_esophagitis","Hyperlipidemia", "Anxiety", "Soft_drink", "City", "Job","ReAdmis")

# Combine the lists of variables
variables_to_keep <- c(continuous_variables, categorical_variables)

# Subset the dataframe to only include these variables
medical_data <- medical_data[, variables_to_keep]


summary_categorical <- summary(medical_data[categorical_variables])

# Summary statistics for continuous variables
summary_continuous <- summary(medical_data[continuous_variables])



# Univariate Statistics for Continuous variables. Function to create and save histogram for each variable
create_histogram <- function(data, variable_name) {
  p <- ggplot(data, aes_string(x = variable_name)) +
    geom_histogram(binwidth = 30, color = "black", fill = "blue") +
    ggtitle(paste("Histogram of", variable_name))
  
  # Construct the full file path
  file_path <- paste0("/Users/john/Downloads/", variable_name, "_histogram.png")
  
  # Save the plot
  ggsave(file_path, plot = p, width = 10, height = 8)
}

# Loop through each continuous variable in continuous_variables and create a histogram
for (variable_name in continuous_variables) {
  create_histogram(medical_data, variable_name)
}


# Univariate Statistic for Categorical Variables - Loop through each categorical variable to create a bar chart
for (variable in categorical_variables) {
  # Open PNG device
  png(file = paste0("/Users/john/Downloads/bar_chart_", variable, ".png"))
  
  plot <- ggplot(medical_data, aes_string(x=variable)) +
    geom_bar(fill="steelblue", color="black") +
    labs(title=paste("Count of Records by", variable), x=variable, y="Count") +
    theme_minimal() +
    theme(axis.text.x=element_text(angle=45, hjust=1))
  print(plot)
  
  # Close PNG device
  dev.off()
}

# Bivariate Statistics - Continuous vs Categorical use Grouped Boxplots
for (variable in continuous_variables) {
  # Open PNG device
  png(file = paste0("C:/Users/john/Downloads/grouped_boxplot_", variable, ".png"))
  
  plot <- ggplot(medical_data, aes_string(x = variable, y = "ReAdmis", group = variable)) +
    geom_boxplot() +
    labs(title = paste("ReAdmis Distribution across", variable), x = variable, y = "ReAdmis") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(plot) # Display the plot
  
  # Close PNG device
  dev.off()
}

# Bivariate Statistics - Categorical vs Categorical use Two-way Frequency tables

# Load necessary libraries
library(ggplot2)


# Loop through each categorical variable
freq_table <- table(medical_data[, categorical_variables[1]], 
                    medical_data[, categorical_variables[2]], 
                    medical_data$ReAdmis)

# Using ftable() function


# Assuming medical_data is already defined and contains 'ReAdmis'

# Load necessary libraries
library(ggplot2)

# Loop through each pair of categorical variables
for (i in 1:(length(categorical_variables)-1)) {

    
    # Create the frequency table for the current pair of variables
    freq_table <- ftable(medical_data[, categorical_variables[i]], 
                         medical_data$ReAdmis)
    
    # Convert the frequency table to a dataframe for plotting
    freq_df <- as.data.frame(freq_table)
    
    # Create the plot
    p <- ggplot(freq_df, aes(x = Var1, y = Freq, fill = "ReAdmis")) + 
      geom_bar(stat = "identity", position = "dodge") + 
      facet_wrap(~Var2, scales = "free") + 
      theme_minimal() + 
      labs(x = categorical_variables[i], y = "Frequency", fill = "ReAdmis")
    
    # Define the filename based on the variables
    file_name <- paste("freq_table_", categorical_variables[i], "_", ".png", sep = "")
    
    # Export the plot to a PNG file
    png(file_name, width = 800, height = 600)
    print(p)
    dev.off()
  
}


#Convert categorical data points to 1 and 0
i
medical_data[categorical_variables] <- lapply(medical_data[categorical_variables], factor)

# Now, generate the summary specifically for these variables
categorical_summary <- dfSummary(medical_data[categorical_variables], style = "grid")

summary_categorical <- summary(medical_data[categorical_variables])

# Summary statistics for continuous variables
summary_continuous <- summary(medical_data[continuous_variables])
summary_continuous
summary_categorical
categorical_summary
#Create Dummy Variables for each Categorical data https://www.bzst.com/2015/08/categorical-predictors-how-many-dummies.html

# Create dummy variables for each categorical variable except "ReAdmis"
categorical_variables <- c("Reflux_esophagitis","Hyperlipidemia", "Anxiety", "Soft_drink", "City", "Job")


# Create dummy variables for each categorical variable
dummy_variables <- list()

for (cat_var in categorical_variables) {
  # Ensure the variable is treated as a factor
  medical_data[[cat_var]] <- factor(medical_data[[cat_var]])
  
  # Check if the variable has at least two levels
  if (length(levels(medical_data[[cat_var]])) < 2) {
    next  # Skip this iteration if the variable doesn't have at least two levels
  }
  
  # Create dummy variables using model.matrix
  dummies <- model.matrix(~ 0 + medical_data[[cat_var]])
  
  # Get the level names directly from the factor
  level_names <- levels(medical_data[[cat_var]])
  
  # Generate unique names based on the original variable name and level names
  # Remove non-letter characters from names
  cleaned_names <- gsub("[^A-Za-z]", "", paste0(cat_var, "_", level_names))
  
  # Assign cleaned, valid names to the columns of dummies
  colnames(dummies) <- cleaned_names
  
  # Store dummy variables in the list
  dummy_variables[[cat_var]] <- dummies
}

# Combine all dummy variables into a single data frame
dummy_data <- as.data.frame(do.call(cbind, dummy_variables))

# Ensure column names are valid variable names without spaces or other special characters
names(dummy_data) <- make.names(names(dummy_data), unique = TRUE)

#Combine continuous and dummy variables into medical_data
medical_data <- medical_data %>%
  mutate(ReAdmis = ifelse(ReAdmis == "Yes", 1, 0))

medical_data_updated <- cbind(medical_data[continuous_variables], dummy_data)
# Assuming medical_data_updated already exists and contains data
medical_data_updated <- cbind(medical_data_updated, ReAdmis = medical_data$ReAdmis)
medical_data <- medical_data_updated

#Output the cleaned dataset as a csv file
medical_data_cleaned <- medical_data
file_path_cleanedData <- "/Users/john/Downloads/medical_data_cleaned_submission.csv"
write.csv(medical_data_cleaned, file_path_cleanedData, row.names = FALSE)

#-----------------------------------------Model Syntax--------------

# Create the logistic regression model
logistic_model <- glm(ReAdmis ~ ., data = medical_data, family = binomial(link = "logit"), maxit = 1000)

coefficients <- coef(logistic_model)

# Make predictions on the dataset (return probabilities)
probabilities <- predict(logistic_model, type = "response")

# Convert probabilities to binary outcomes based on a threshold (e.g., 0.5)
predicted_outcomes <- factor(ifelse(probabilities > 0.7, 1, 0))
actual_outcomes <- factor(medical_data$ReAdmis)

# Get unique levels from both predicted and actual outcomes
all_levels <- c(1, 0)

# Refactor levels of both variables to match
predicted_outcomes <- factor(predicted_outcomes, levels = all_levels)
actual_outcomes <- factor(actual_outcomes, levels = all_levels)

# Calculate confusion matrix
confusion_matrix <- confusionMatrix(predicted_outcomes, actual_outcomes)

# Extracting the overall accuracy from the confusion matrix
accuracy <- confusion_matrix$overall[["Accuracy"]]

# Output the confusion matrix
confusion_matrix <- confusion_matrix$table

#Reformat Confusion Matrix in a readable format
cm_table <- confusion_matrix
logistic_model_cm <- data.frame(
  'Predicted Negative' = c(cm_table[1,1], cm_table[2,1]), # True Negative and False Negative
  'Predicted Positive' = c(cm_table[1,2], cm_table[2,2]), # False Positive and True Positive
  row.names = c('Actual Negative', 'Actual Positive')
)

logistic_model_cm

# To extract p-values of model coefficients
p_values_logistic_model <- summary(logistic_model)$coefficients[, "Pr(>|z|)"]

# Output the accuracy
accuracy

# Create equation for initial logistic regression model
model_coefficients <- coef(logistic_model)

# Begin constructing the equation string
equation_str <- "P = exp(Logit(P)) / (1 + exp(Logit(P)))\nwhere Logit(P) = "

# Add the intercept
equation_str <- paste(equation_str, model_coefficients[1])

# Loop through remaining coefficients to construct the equation
for (i in 2:length(model_coefficients)) {
  # Add each term
  equation_str <- paste(equation_str, " + (", model_coefficients[i], " * ", names(model_coefficients)[i], ")", sep="")
}

# For logistic regression, it's common to use the predicted probabilities of the positive class
#Optional: Rename the columns in medical_data the same as the name as the coefficients from reduced_logistic_model
predictions_logistic_model <- predict(logistic_model, newdata = medical_data, type = "response")

#----------------------------------Summary for the Logistic Model
# Extract model summary
model_summary <- summary(logistic_model)

# Coefficients
coefficients <- model_summary$coefficients[, "Estimate"]

# Standard Errors
standard_errors <- model_summary$coefficients[, "Std. Error"]

# z-values
z_values <- model_summary$coefficients[, "z value"]

# P-values
p_values <- model_summary$coefficients[, "Pr(>|z|)"]

# Variable names
variable_names <- rownames(model_summary$coefficients)

# Combine into a dataframe
model_summary <- data.frame(Variable = variable_names, 
                       Coefficients = coefficients, 
                       Std_Error = standard_errors, 
                       Z_value = z_values,
                       P_value = p_values)

# Define file path
file_path <- "logistic_model_summary.csv"
model_summary
# Export dataframe as CSV
write.csv(model_summary, file_path, row.names = FALSE)

#Reducition of initial Logistic Regression Model

#Recusrive feature Sselection 
#-------------------------------------------------------------------------

# Prepare the data
set.seed(123) # Set seed for reproducibility
data <- medical_data # Assuming medical_data is your dataframe

# Ensure that the dependent variable is a factor for logistic regression
data$ReAdmis <- as.factor(data$ReAdmis)

# Create a control function to use within train with method = "rfe"
# This example uses logistic regression (method = "glm") for binary classification
control <- rfeControl(functions = rfFuncs, # rfFuncs can be used for classification tasks
                      method = "cv", # Cross-validation
                      number = 10) # Number of folds in cross-validation

# Specify the outcome and predictor variables
outcomeName <- 'ReAdmis'
predictors <- setdiff(names(data), outcomeName)
sizesToExplore <- c(1:10, seq(20, 100, by = 20), seq(150, 650, by = 100))
# Run Recursive Feature Elimination
set.seed(123) # For reproducibility
rfeResults <- rfe(data[, predictors],
                  data[, outcomeName],
                  sizes = sizesToExplore, # Adjust based on the number of predictors you wish to evaluate
                  rfeControl = control,
                  method = "glm", # Logistic regression
                  family = "binomial") # Since it's a binary classification

# View results
print(rfeResults)

# To see the ranking of variables
print(rfeResults$variables)

finalVariables <- predictors(rfeResults)
# Add the dependent variable name to the list if it's not already included
allVariables <- c(finalVariables, "ReAdmis")

# Subset the dataframe to keep only the selected predictors and the dependent variable
medical_data <- medical_data[, allVariables]

#----------------------------- Reduced Model----------------------------------------------------------------
# Fit the reduced logistic regression model
reduced_logistic_model <- glm(ReAdmis ~ ., data = medical_data, family = binomial(link = "logit"), maxit = 1000)

reduced_coefficients <- coef(reduced_logistic_model)

# Make predictions on the dataset (return probabilities)
reduced_probabilities <- predict(reduced_logistic_model, type = "response")

# Convert probabilities to binary outcomes based on a threshold (e.g., 0.5)
reduced_predicted_outcomes <- factor(ifelse(reduced_probabilities > 0.5, 1, 0))
reduced_actual_outcomes <- factor(medical_data$ReAdmis)

# Get unique levels from both predicted and actual outcomes
reduced_all_levels <- c(1, 0)

# Refactor levels of both variables to match
reduced_predicted_outcomes <- factor(reduced_predicted_outcomes, levels = all_levels)
reduced_actual_outcomes <- factor(reduced_actual_outcomes, levels = all_levels)

# Calculate confusion matrix
reduced_confusion_matrix <- confusionMatrix(reduced_predicted_outcomes, reduced_actual_outcomes)


# Extracting the overall accuracy from the confusion matrix
reduced_accuracy <- reduced_confusion_matrix$overall[["Accuracy"]]

# Output the confusion matrix
reduced_confusion_matrix <- reduced_confusion_matrix$table

#Reformat Confusion Matrix in a readable format
reduced_cm_table <- reduced_confusion_matrix
reduced_logistic_model_cm <- data.frame(
  'Predicted Negative' = c(reduced_cm_table[1,1], reduced_cm_table[2,1]), # True Negative and False Negative
  'Predicted Positive' = c(reduced_cm_table[1,2], reduced_cm_table[2,2]), # False Positive and True Positive
  row.names = c('Actual Negative', 'Actual Positive')
)

reduced_logistic_model_cm

# To extract p-values of model coefficients
reduced_p_values_logistic_model <- summary(reduced_logistic_model)$coefficients[, "Pr(>|z|)"]

# Output the accuracy
reduced_accuracy

# Create equation for initial logistic regression model
reduced_model_coefficients <- coef(reduced_logistic_model)

# Begin constructing the equation string
reduced_equation_str <- "P = exp(Logit(P)) / (1 + exp(Logit(P)))\nwhere Logit(P) = "

# Add the intercept
reduced_equation_str <- paste(reduced_equation_str, reduced_model_coefficients[1])

# Loop through remaining coefficients to construct the equation
for (i in 2:length(reduced_model_coefficients)) {
  # Add each term
  reduced_equation_str <- paste(reduced_equation_str, " + (", reduced_model_coefficients[i], " * ", names(reduced_model_coefficients)[i], ")", sep="")
}

#----------------------------------Summary for the Logistic Model
# Extract model summary
reduced_model_summary <- summary(reduced_logistic_model)

# Coefficients
reduced_coefficients <- reduced_model_summary$coefficients[, "Estimate"]

# Standard Errors
reduced_standard_errors <- reduced_model_summary$coefficients[, "Std. Error"]

# z-values
reduced_z_values <- reduced_model_summary$coefficients[, "z value"]

# P-values
reduced_p_values <- reduced_model_summary$coefficients[, "Pr(>|z|)"]

# Variable names
reduced_variable_names <- rownames(reduced_model_summary$coefficients)

# Combine into a dataframe
reduced_model_summary <- data.frame(Variable = reduced_variable_names, 
                            Coefficients = reduced_coefficients, 
                            Std_Error = reduced_standard_errors, 
                            Z_value = reduced_z_values,
                            P_value = reduced_p_values)

# Define file path
file_path <- "reduced_logistic_model_summary.csv"
reduced_model_summary
# Export dataframe as CSV
write.csv(reduced_model_summary, file_path, row.names = FALSE)

#-----------------------------ROC Curves------------------------------------------------------------
#Compare both model performances using a ROC Curve
# Load the pROC package
library(pROC)

# First, generate predictions from both models

# For logistic regression, it's common to use the predicted probabilities of the positive class
#Optional: Rename the columns in medical_data the same as the name as the coefficients from reduced_logistic_model
predictions_reduced_logistic_model <- predict(reduced_logistic_model, newdata = medical_data, type = "response")

# Now, compute the ROC curve and AUC for both sets of predictions
roc_logistic_model <- roc(medical_data$ReAdmis, predictions_logistic_model)
roc_reduced_logistic_model <- roc(medical_data$ReAdmis, predictions_reduced_logistic_model)

# Compare the AUC values
auc_logistic_model <- auc(roc_logistic_model)
auc_reduced_logistic_model <- auc(roc_reduced_logistic_model)

# Print the AUC values for comparison
cat("AUC for logistic_model: ", auc_logistic_model, "\n")
cat("AUC for reduced_logistic_model: ", auc_reduced_logistic_model, "\n")

file_path <- "C:/Users/John/Desktop"

# Open a PNG graphics device for the original model's ROC curve
# Plot ROC curve for the original logistic model
png("original_logistic_model_ROC_curve.png", width = 800, height = 600)
plot(roc_logistic_model, col = "blue", main = "ROC Curve for Original Logistic Model")
legend("bottomright", legend = c("Original Model"), col = c("blue"), lwd = 2)
dev.off()

# Plot ROC curve for the reduced logistic model
png("reduced_logistic_model_ROC_curve.png", width = 800, height = 600)
plot(roc_reduced_logistic_model, col = "red", main = "ROC Curve for Reduced Logistic Model")
legend("bottomright", legend = c("Reduced Model"), col = c("red"), lwd = 2)
dev.off()

# Move the saved files to the specified file path
file.copy(from = "original_logistic_model_ROC_curve.png", 
          to = file.path(file_path, "original_logistic_model_ROC_curve.png"), 
          overwrite = TRUE)
file.copy(from = "reduced_logistic_model_ROC_curve.png", 
          to = file.path(file_path, "reduced_logistic_model_ROC_curve.png"), 
          overwrite = TRUE)

# Lastly output the calculations and metrics of each logistic model 
auc_logistic_model
auc_reduced_logistic_model
logistic_model_cm
reduced_logistic_model_cm
p_values_logistic_model
reduced_p_values_logistic_model
probabilities
reduced_probabilities
accuracy
reduced_accuracy
model_coefficients
reduced_model_coefficients
model_summary
reduced_model_summary


