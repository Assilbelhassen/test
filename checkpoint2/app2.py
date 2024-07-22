import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier



# Load the dataset
file_path =  r'C:\Users\User\Downloads\Financial_inclusion_dataset.csv'  
data = pd.read_csv(file_path)

# Display general information about the dataset
print(data.info())
print(data.head())

# Handle missing values
data_cleaned = data.dropna()

# Remove duplicates
data_cleaned = data_cleaned.drop_duplicates()

# Display cleaned data info
print(data_cleaned.info())
print(data_cleaned.head())

# Encode categorical features
data_encoded = pd.get_dummies(data_cleaned, drop_first=True)

# Display encoded data info
print(data_encoded.info())
print(data_encoded.head())

# Define the target variable and features
# Replace 'bank_account' with the actual column name for the target variable if different
target_column_name = 'bank_account'

X = data_encoded.drop(target_column_name + '_Yes', axis=1)
y = data_encoded[target_column_name + '_Yes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')



# Assuming `clf` is your trained Random Forest model
model_filename = 'random_forest_model.pkl'
joblib.dump(clf, model_filename)


# Load the trained model
model_filename = 'random_forest_model.pkl'
clf = joblib.load(model_filename)

# Define the input fields
st.title("Financial Inclusion Prediction")

st.write("Please provide the following information:")

# Add input fields for each feature
country = st.selectbox('Country', ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
year = st.selectbox('Year', [2016, 2017, 2018])
uniqueid = st.text_input('Unique ID')
location_type = st.selectbox('Location Type', ['Urban', 'Rural'])
cellphone_access = st.selectbox('Cellphone Access', ['Yes', 'No'])
household_size = st.number_input('Household Size', min_value=1, max_value=20, value=1)
age_of_respondent = st.number_input('Age of Respondent', min_value=15, max_value=100, value=25)
gender_of_respondent = st.selectbox('Gender of Respondent', ['Male', 'Female'])
relationship_with_head = st.selectbox('Relationship with Head', ['Head of Household', 'Spouse', 'Child', 'Parent', 'Other relative', 'Other non-relatives'])
marital_status = st.selectbox('Marital Status', ['Married/Living together', 'Divorced/Seperated', 'Widowed', 'Single/Never Married'])
education_level = st.selectbox('Education Level', ['No formal education', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training'])
job_type = st.selectbox('Job Type', ['Government Dependent', 'Self employed', 'Formally employed Private', 'Formally employed Government', 'Informally employed', 'Employer', 'Unemployed', 'Student', 'Retired'])

# Validation button
if st.button('Predict'):
    # Encode the input values similarly to the training data
    input_data = pd.DataFrame({
        'country': [country],
        'year': [year],
        'uniqueid': [uniqueid],
        'location_type': [location_type],
        'cellphone_access': [cellphone_access],
        'household_size': [household_size],
        'age_of_respondent': [age_of_respondent],
        'gender_of_respondent': [gender_of_respondent],
        'relationship_with_head': [relationship_with_head],
        'marital_status': [marital_status],
        'education_level': [education_level],
        'job_type': [job_type]
    })

    # One-hot encode the input data to match the model's training format
    input_data_encoded = pd.get_dummies(input_data)
    
    # Ensure that all necessary columns are present
    missing_cols = set(X.columns) - set(input_data_encoded.columns)
    for col in missing_cols:
        input_data_encoded[col] = 0
    input_data_encoded = input_data_encoded[X.columns]
    
    # Predict using the loaded model
    prediction = clf.predict(input_data_encoded)

    # Display the prediction
    if prediction[0] == 1:
        st.success("The respondent is likely to have a bank account.")
    else:
        st.error("The respondent is unlikely to have a bank account.")

