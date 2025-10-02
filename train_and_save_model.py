import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

# --- 1. Define Filepaths ---
MODEL_FILE = 'stacking_model.pkl'
SCALER_FILE = 'scaler.pkl'
TARGET_ENCODER_FILE = 'target_encoder.pkl'
FEATURE_ENCODERS_FILE = 'feature_encoders.pkl'

# --- 2. Define Features and Column Renaming ---

# **CRITICAL UPDATE:** Mapped missing features based on the CSV content.
COLUMN_MAPPING = {
    # Existing Mappings (Confirmed)
    'Study_Hours_per_Week': 'Study_Time_Weekly',
    'Parent_Education_Level': 'Parental_Education',
    'Family_Income_Level': 'Financial_Support',
    'Stress_Level (1-10)': 'Stress_Level',
    'Attendance (%)': 'Attendance',
    'Sleep_Hours_per_Night': 'Sleep_Hours',
    
    # NEW Mappings based on the missing columns reported in the last error:
    # 'Test_Score' is already a column, but 'Midterm_Score' and 'Final_Score' were dropped. 
    # Let's use the 'Total_Score' as the primary performance metric if 'Test_Score' is generic/missing.
    # However, since 'Test_Score' was explicitly listed as missing, we must keep it in the list of expected columns.
    # NOTE: Since your CSV doesn't have 'Ethnicity' or 'Tutoring', we will use the existing columns that are related to performance.
    
    # We will remove the features that are genuinely missing from the training lists and adjust the remaining ones.
}

# The model will use these simplified names for consistency:
# Based on your CSV, 'Ethnicity' and 'Tutoring' are NOT present as single columns.
# We must remove them from the list of required features.
# FIX: Added 'Internet_Access_at_Home' to prevent the ValueError: 'No'
CATEGORICAL_FEATURES = ['Gender', 'Extracurricular_Activities', 'Parental_Education', 'Financial_Support', 'Internet_Access_at_Home']
NUMERICAL_FEATURES = ['Age', 'Study_Time_Weekly', 'Attendance', 
                      'Test_Score', 'Performance_Index', 'Sleep_Hours', 
                      'Sample_Question_Papers_Practiced']
TARGET_FEATURE = 'Stress_Level'

# Columns to drop from the dataset (Removed 'Ethnicity' and 'Tutoring' from the drop list)
COLUMNS_TO_DROP = ["Student_ID", "First_Name", "Last_Name", "Email", 
                   "Department", "Total_Score", "Grade", 
                   "Midterm_Score", "Final_Score", "Assignments_Avg", 
                   "Quizzes_Avg", "Participation_Score", "Projects_Score"] 

# Since the previous run showed 'Ethnicity', 'Tutoring', 'Test_Score', 'Performance_Index', 
# and 'Sample_Question_Papers_Practiced' were missing, let's verify if they exist in the CSV.
# Your CSV headers are: Student_ID,First_Name,Last_Name,Email,Gender,Age,Department,Attendance (%),Midterm_Score,Final_Score,Assignments_Avg,Quizzes_Avg,Participation_Score,Projects_Score,Total_Score,Grade,Study_Hours_per_Week,Extracurricular_Activities,Internet_Access_at_Home,Parent_Education_Level,Family_Income_Level,Stress_Level (1-10),Sleep_Hours_per_Night

# Based on your CSV, these five columns are NOT present with the required names.
# We need to use columns that ARE present. Let's adjust the NUMERICAL_FEATURES list to use 
# the scores that exist and assume 'Performance_Index' and 'Sample_Question_Papers_Practiced' 
# are not available in this version of the data.

NUMERICAL_FEATURES = ['Age', 'Study_Time_Weekly', 'Attendance', 'Sleep_Hours']
# Adding score-related columns that exist and are not dropped
NUMERICAL_FEATURES += ['Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score', 'Projects_Score']

# Let's adjust the drop list to keep the score-related numerical features
COLUMNS_TO_DROP = ["Student_ID", "First_Name", "Last_Name", "Email", 
                   "Department", "Total_Score", "Grade"] 

# FINAL LISTS AFTER CAREFUL REVIEW OF YOUR CSV:
# CRITICAL FIX: 'Internet_Access_at_Home' is now included here.
CATEGORICAL_FEATURES = ['Gender', 'Extracurricular_Activities', 'Parental_Education', 'Financial_Support', 'Internet_Access_at_Home']
NUMERICAL_FEATURES = ['Age', 'Study_Time_Weekly', 'Attendance', 'Sleep_Hours', 
                      'Midterm_Score', 'Final_Score', 'Assignments_Avg', 
                      'Quizzes_Avg', 'Participation_Score', 'Projects_Score']
TARGET_FEATURE = 'Stress_Level'
# NOTE: Removed 'Ethnicity', 'Tutoring', 'Test_Score', 'Performance_Index', and 'Sample_Question_Papers_Practiced'
# as they do not exist in your dataset. The Streamlit app will need to be updated later to match these inputs.


# --- 3. Load Data and Preprocessing ---
try:
    # Assuming the file is in the same directory as this script.
    df = pd.read_csv("Students Performance Dataset.csv") 
    
except FileNotFoundError:
    print("FATAL ERROR: 'Students Performance Dataset.csv' not found.")
    print("Please ensure the dataset file is in the same directory as this script.")
    exit()

# A. RENAME COLUMNS to standardized model names
df.rename(columns=COLUMN_MAPPING, inplace=True)
print("Columns renamed successfully.")

# B. Drop unnecessary columns
existing_cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
if existing_cols_to_drop:
    df.drop(existing_cols_to_drop, axis=1, inplace=True, errors='ignore')
    print(f"Dropped non-feature columns: {existing_cols_to_drop}")
else:
    print("No non-feature columns found to drop.")


# C. Check if all expected columns are present
all_expected_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_FEATURE]
missing_cols = [col for col in all_expected_cols if col not in df.columns]

if missing_cols:
    print(f"FATAL ERROR: Missing required columns in the dataset: {missing_cols}")
    print("\nACTION REQUIRED: Please verify the names in the COLUMN_MAPPING list.")
    exit()
else:
    print("All required columns found.")


# --- 4. Preprocessing: Encoding and Scaling ---

# A. Encode the Target Variable (Stress_Level)
target_encoder = LabelEncoder()
df[TARGET_FEATURE] = target_encoder.fit_transform(df[TARGET_FEATURE])
joblib.dump(target_encoder, TARGET_ENCODER_FILE)
print(f"Target Encoder saved to {TARGET_ENCODER_FILE}")

# B. Encode Categorical Features (X)
feature_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    # Handle potential NaNs by converting to string before fitting
    df[col] = df[col].astype(str) 
    df[col] = le.fit_transform(df[col])
    feature_encoders[col] = le
joblib.dump(feature_encoders, FEATURE_ENCODERS_FILE)
print(f"Feature Encoders saved to {FEATURE_ENCODERS_FILE}")


# C. Standardize Numerical Features (X)
scaler = StandardScaler()
df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])
joblib.dump(scaler, SCALER_FILE)
print(f"Scaler saved to {SCALER_FILE}")


# --- 5. Model Training ---

X = df.drop(TARGET_FEATURE, axis=1)
y = df[TARGET_FEATURE]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define base estimators for the Stacking Classifier 
estimators = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', XGBClassifier(random_state=42))
]

# Define the Stacking Classifier
stacked_model = StackingClassifier(
    estimators=estimators, 
    final_estimator=LogisticRegression(solver='liblinear'), 
    cv=5
)

print("\nStarting Stacking Model training...")
stacked_model.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Save the Trained Model ---
joblib.dump(stacked_model, MODEL_FILE)
print(f"Trained Stacking Classifier model saved to {MODEL_FILE}")

# --- SUCCESS CHECK ---
if all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, TARGET_ENCODER_FILE, FEATURE_ENCODERS_FILE]):
    print("\n✅ SUCCESS: All four required model files have been created.")
    print("You can now safely run 'streamlit run streamlit_app.py'.")
else:
    print("\n❌ FAILURE: One or more model files were not saved. Check for previous errors.")
