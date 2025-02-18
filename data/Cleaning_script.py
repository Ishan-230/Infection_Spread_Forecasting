import pandas as pd

# Load the datasets
dengue_data = pd.read_csv('dengue_cases_in_india.csv')
typhoid_data = pd.read_csv('typhoid.csv')

# Step 1: Standardize column names (lowercase, remove spaces)
dengue_data.columns = dengue_data.columns.str.strip().str.lower()
typhoid_data.columns = typhoid_data.columns.str.strip().str.lower()

# Step 2: Convert 'date' columns to datetime format (if they exist)
if 'date' in dengue_data.columns:
    dengue_data['date'] = pd.to_datetime(dengue_data['date'], errors='coerce')
if 'date' in typhoid_data.columns:
    typhoid_data['date'] = pd.to_datetime(typhoid_data['date'], errors='coerce')

# Step 3: Remove duplicate rows if any
dengue_data = dengue_data.drop_duplicates()
typhoid_data = typhoid_data.drop_duplicates()

# Step 4: Handle missing values by removing rows with missing data
dengue_data = dengue_data.dropna()
typhoid_data = typhoid_data.dropna()

# Step 5: Save the cleaned datasets to new CSV files
dengue_data.to_csv('C:\\Users\\priya\\OneDrive\\Desktop\\chat gpt\\cleaned_dengue_data.csv', index=False)
typhoid_data.to_csv('C:\\Users\\priya\\OneDrive\\Desktop\\chat gpt\\cleaned_typhoid_data.csv', index=False)

print("Data cleaning complete. Cleaned files saved.")
