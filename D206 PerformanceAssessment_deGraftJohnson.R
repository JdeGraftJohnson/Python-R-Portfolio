# D206 - Data Acquisition

#rm(list = ls()) Used to clear the environment

# Install and load relevant packages if not already installed
if (!require(dplyr)) install.packages("dplyr")
library(dplyr)

if (!require("stats")) install.packages("stats")
library(stats)

install.packages("tidyverse")
library(purrr)

# Define the correct file path to medical_raw_data.csv
file_path <- "/Users/jdegraft/Downloads/medical_raw_data.csv"

# Read the CSV file into a variable (e.g., 'medical_data')
medical_data <- read.csv(file_path)

# View the structure or summary of the imported data
str(medical_data)
summary(medical_data)

#Create a histogram to view data distribution for each numeric column

# Identify numeric columns
numeric_columns <- sapply(medical_data, is.numeric)

# Loop through each numeric column and plot a histogram to observe outliers
photo_path = "/Users/jdegraft/Downloads"

for (col in names(numeric_columns[numeric_columns])) {
  # Plot histogram
  png(filename = file.path(photo_path, paste("histogram_", col, ".png", sep = ""))) # Define the file path
  hist(medical_data[[col]], main = paste("Histogram of", col), xlab = col)
  dev.off() # Close the PNG device
}

# Check duplicates based on specific columns, e.g., 'column1' and 'column2'
specific_duplicate_rows <- medical_data[duplicated(medical_data[c("Customer_id")]),]

# Convert Anxiety & Overweight Columns into categorical to ensure it is one hot encoded with the rest
medical_data$Anxiety <- ifelse(medical_data$Anxiety == 1, "Yes", "No")
medical_data$Overweight <- ifelse(medical_data$Overweight == 1, "Yes", "No")

#Check and keep count of any NULL or blank rows in the medical_data dataframe
blank_count <- sum(sapply(medical_data, function(column) {
  if (is.character(column)) {
    sum(column == "")
  } else {
    0
  }
}))

# Check for NULLs in list columns (less common in dataframes)
null_count <- sum(sapply(medical_data, function(column) {
  if (is.list(column)) {
    sum(sapply(column, is.null))
  } else {
    0
  }
}))

# Keep a list of column names with NAs and their count
columns_with_NAs <- list()

# Iterate through columns in the dataset
for (col_name in names(medical_data)) {
  
  # Check if the column contains NAs
  if (anyNA(medical_data[[col_name]])) {
    NA_count <- sum(is.na(medical_data[[col_name]]))
    columns_with_NAs[[col_name]] <- NA_count
    
    # Check if the column is numeric
    if (is.numeric(medical_data[[col_name]])) {
      # Replace NAs with the mean of the column
      mean_val <- mean(medical_data[[col_name]], na.rm = TRUE)
      medical_data[[col_name]][is.na(medical_data[[col_name]])] <- mean_val
    } else {
      # Column is categorical; replace NAs with the mode
      mode_val <- names(sort(table(medical_data[[col_name]]), decreasing = TRUE))[1]
      medical_data[[col_name]][is.na(medical_data[[col_name]])] <- mode_val
    }
  }
}

# Export cleaned dataset
file_path_cleanedData <- "/Users/jdegraft/Downloads/medical_data_cleaned.csv"
write.csv(medical_data, file_path_cleanedData, row.names = FALSE)

# Columns to exclude from Z-score calculation
exclude_cols <- c('X', 'CaseOrder', 'Customer_id', 'Interaction', 'UID', 'Zip', 'Lat', 'Lng')

# Creating a new dataframe with only the columns in exclude_cols
medical_data_excluded <- medical_data[, exclude_cols]

#Update the medical_data dataframe to exclude those columns
medical_data <- medical_data[, !names(medical_data) %in% exclude_cols]

# Calculate Z-scores for each numeric column while ignoring the rows with NA
medical_data <- medical_data %>%
  mutate_if(is.numeric, ~{
    z_scores = (. - mean(., na.rm = TRUE)) / sd(., na.rm = TRUE)
    z_scores
  })

# Count rows with absolute z-scores greater than 3 in each numeric column
high_z_score_counts <- medical_data %>%
  select_if(is.numeric) %>%
  map(~sum(abs(.) > 3)) %>%
  discard(~. == 0) # Optionally, remove columns with zero counts

# Store Z scores with absolute values of 3 or below in medical_data_constrained
medical_data_constrained <- medical_data

# Loop through numeric columns
numeric_cols <- sapply(medical_data_constrained, is.numeric)
for (col in names(medical_data_constrained)[numeric_cols]) {
  medical_data_constrained <- subset(medical_data_constrained, abs(medical_data_constrained[[col]]) <= 3)
}
  
#Export cleaned data set with Z scores - Used for debugging
file_path_cleanedData <- "/Users/jdegraft/Downloads/medical_data_ZScore.csv"
write.csv(medical_data_constrained, file_path_cleanedData, row.names = FALSE)

# Prior to running Principal Component Analysis...ensure categorical variables are ignored
# Assuming you have a list of column names that are numeric
numeric_cols <- c(
  "Population", "Children", "Age", "Income", "VitD_levels", "Doc_visits",
  "Full_meals_eaten", "VitD_supp", "Initial_days", "TotalCharge",
  "Additional_charges", "Item1", "Item2", "Item3", "Item4", "Item5",
  "Item6", "Item7", "Item8"
)

# Identify and store the numeric columns in medical_data_numeric
medical_data_numeric <- medical_data_constrained[numeric_cols]

# Identify and store the categorical columns in medical_data_categorical
non_numeric_cols <- sapply(medical_data_constrained, function(col) !is.numeric(col))

# Extract the names of non-numeric columns
medical_data_categorical <- names(medical_data_constrained)[non_numeric_cols]

# Run PCA
pca_result <- prcomp(medical_data_numeric, center = TRUE, scale. = TRUE)

# View summary of PCA results
summary(pca_result)

# Access the principal components
pca_components <- pca_result$x

# You can also view the loadings
pca_loadings <- pca_result$rotation

pca_loadings

# Calculate variance explained by each principal component
var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)

cumulative_variance <- cumsum(var_explained)
num_components <- which(cumulative_variance >= 0.9)[1] #Outputs number of PCA components based on the smallest number of components needed to explain at least 90% of the total variance

# Create a scree plot
plot(var_explained, xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", 
     type = "b", pch = 19, main = "Scree Plot")

# Export all plots
plots.dir.path <- list.files(tempdir(), pattern="rs-graphics", full.names = TRUE); 
plots.png.paths <- list.files(plots.dir.path, pattern=".png", full.names = TRUE)

# Define the correct file path to save the images
file_path2 <- "/Users/jdegraft/Downloads"

file.copy(from=plots.png.paths, to=file_path2)