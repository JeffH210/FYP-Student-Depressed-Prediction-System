import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Feature Selection
def select_features(df):
    selected_features = [
        'Gender', 'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction',
        'Sleep Duration', 'Dietary Habits',
        'Have you ever had suicidal thoughts ?',
        'Work/Study Hours', 'Financial Stress',
        'Family History of Mental Illness', 'Depression'
    ]
    df = df[selected_features].copy()
    df = df.rename(columns={
        'Work/Study Hours': 'Study Hours',
        'Have you ever had suicidal thoughts ?': 'Suicidal Thoughts'
    })
    return df

# Step 2: Handle Missing data and numeric data
def clean_numeric_data(df):
    df['Financial Stress'] = pd.to_numeric(df['Financial Stress'], errors='coerce')
    df['Financial Stress'].fillna(df['Financial Stress'].mean(), inplace=True)
    df['Financial Stress'] = df['Financial Stress'].astype(int)
    df.dropna(inplace=True)
    df['CGPA'] = (df['CGPA'] / 10) * 4
    df['CGPA'] = df['CGPA'].round(2)
    return df

# Step 3: Remove Outliers using IQR
def remove_outliers(df, columns=['Age', 'CGPA']):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Step 4: Encode categorical variables
def encode_categorical(df):
    le = LabelEncoder()
    for col in ['Gender', 'Suicidal Thoughts', 'Family History of Mental Illness']:
        df[col] = le.fit_transform(df[col])
    
    sleep_map = {
        "'Less than 5 hours'": 0,
        "'5-6 hours'": 1,
        "'7-8 hours'": 2,
        "'More than 8 hours'": 3,
        'Others': 4
    }
    df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)

    diet_map = {
        'Healthy': 0,
        'Moderate': 1,
        'Unhealthy': 2,
        'Others': 3
    }
    df['Dietary Habits'] = df['Dietary Habits'].map(diet_map)

    return df

# Step 5: Full preprocessing pipeline
def preprocess_pipeline(df):
    df = select_features(df)
    df = clean_numeric_data(df)
    df = remove_outliers(df)
    df = encode_categorical(df)
    return df

