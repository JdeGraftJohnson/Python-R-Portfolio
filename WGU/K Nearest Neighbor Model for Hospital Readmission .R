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
library(class)


library(pROC)

# Define the correct file path to medical_raw_data.csv
file_path <- "/Users/jdegraft/Downloads/medical_clean.csv"
#file_path <- "C:/Users/John/Desktop/medical_clean.csv"

# Read the CSV file into a variable (e.g., 'medical_data')
medical_data <- read.csv(file_path)


# Filter 'medical_data' to include only the specified states
medical_data <- medical_data %>%
  filter(State %in% c('DC', 'VA', 'DE', 'MD'))
medical_data_csv <- medical_data
#Convert Categorical variables into numerical data points

#Any variable pairings in the significant list should not be included in the research variable list
continuous_variables <- c("Population", "Children", "Age", "VitD_levels", "Doc_visits", "Income","TotalCharge","Initial_days","Additional_charges")

# Null hypothesis - Predictor variables are independent of each other
#Categorical Variables - highest p-values. 10 variables from top_five_info. 
categorical_variables <- c("Reflux_esophagitis","Hyperlipidemia", "Anxiety", "Soft_drink", "City", "Job","ReAdmis")

# Combine the lists of variables
variables_to_keep <- c(continuous_variables, categorical_variables)

# Subset the dataframe to only include these variables
medical_data <- medical_data[, variables_to_keep]


#Create Dummy Variables for each Categorical data https://www.bzst.com/2015/08/categorical-predictors-how-many-dummies.html
install.packages("fastDummies")
library(fastDummies)

# Create dummy variables for each categorical variable except "ReAdmis"
categorical_variables <- c("Reflux_esophagitis","Hyperlipidemia", "Anxiety", "Soft_drink", "City", "Job")

# Assuming 'medical_data' is your dataset and 'categorical_variables' contains the names of the categorical variables

# Create dummy variables for each categorical variable
dummy_variables <- list()

for (cat_var in categorical_variables) {
  # Ensure the variable is treated as a factor
  medical_data[[cat_var]] <- factor(medical_data[[cat_var]])
  
  # Check if the variable has at least two levels
  if (length(levels(medical_data[[cat_var]])) < 2) {
    next  # Skip this iteration if the variable doesn't have at least two levels
  }
  
  # Create dummy variables using model.matrix, including the base level
  # The key change is in the formula passed to model.matrix. By not subtracting 1, we include all levels
  dummies <- model.matrix(~ medical_data[[cat_var]] + 0)
  
  # Adjust the column names to remove the 'medical_data' prefix
  # This step cleans up the variable names to make them more readable and usable
  colnames(dummies) <- gsub("medical_data", "", colnames(dummies))
  
  # The rest of the script remains largely the same, focusing on cleaning and storing the dummy variables
  
  # Get the level names directly from the factor (this step might be redundant with the new approach)
  level_names <- levels(medical_data[[cat_var]])
  
  # Generate unique names based on the original variable name and level names
  # This step might be adjusted or skipped based on the naming convention desired
  cleaned_names <- gsub("[^A-Za-z]", "", paste0(cat_var, "_", level_names))
  
  # Assign cleaned, valid names to the columns of dummies if necessary
  # With the new naming convention, this step might be less critical
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



#-----------------------------------------Model Syntax--------------

# Scale continuous variables
preProcess_range <- preProcess(medical_data[continuous_variables], method = 'range')
medical_data <- predict(preProcess_range, medical_data[continuous_variables])
medical_data$ReAdmis <- medical_data_updated$ReAdmis
# Combine scaled continuous data and dummy variables
final_data <- cbind(medical_data, dummy_data)

#Output the cleaned dataset as a csv file
medical_data_cleaned <- medical_data
file_path_cleanedData <- "/Users/jdegraft/Downloads/medical_data_cleaned_submission.csv"
write.csv(medical_data_cleaned, file_path_cleanedData, row.names = FALSE)

# Split the data into training and testing sets
set.seed(123) # for reproducibility
training_rows <- createDataPartition(final_data$ReAdmis, p = 0.7, list = FALSE)
train_data <- final_data[training_rows, ]
test_data <- final_data[-training_rows, ]

# Implement KNN
# Assuming 'ReAdmis_Yes' is the target variable after dummy encoding
knn_pred <- knn(train = train_data[, -which(names(train_data) == "ReAdmis")], 
                test = test_data[, -which(names(test_data) == "ReAdmis")], 
                cl = train_data$ReAdmis, 
                k = 5)

# Evaluate the Model
confusionMatrix <- table(test_data$ReAdmis, knn_pred)
print(confusionMatrix)
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
print(paste("Accuracy:", accuracy))

#ROC Curve

# Assuming knn_pred contains predicted probabilities for the positive class
# and test_data$ReAdmis contains the actual binary outcomes

# Compute the ROC curve
knn_pred_numeric <- as.numeric(as.character(knn_pred))
roc_obj <- roc(response = test_data$ReAdmis, predictor = knn_pred_numeric)


# Plot the ROC curve
plot(roc_obj, main="ROC Curve for KNN Model")
# Adding AUC to the plot
auc(roc_obj) -> auc_value
legend("bottomright", legend=paste("AUC =", round(auc_value, 2)), 
       box.lty=1, box.lwd=1, cex=0.8, bg='lightblue')

# Print the AUC value
print(auc(roc_obj))


# Display the confusion matrix
print(cm)
