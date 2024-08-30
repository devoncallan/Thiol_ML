
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import itertools
import streamlit as st


def get_gel_data(data_path:str='Gel_data.csv') -> pd.DataFrame:
    
    gel_df = pd.read_csv(data_path)
    return gel_df
    
@st.cache_data()
def train_gel_model(gel_df: pd.DataFrame):
    
    # Splitting the data into features and target
    X = gel_df.drop('Gel', axis=1)
    y = gel_df['Gel']

    # Normalizing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Setting up cross-validation (5 folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create and train the logistic regression model
    model = LogisticRegression()
    detailed_results = cross_validate(model, X_scaled, y, cv=cv, scoring={
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    })
    print(detailed_results)
    
    # Fit the model on the entire scaled dataset for feature importance analysis
    model_full = LogisticRegression().fit(X_scaled, y)

    # Checking the importance of each feature
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model_full.coef_[0]
    })

    # Calculating the absolute value of coefficients to determine impact
    feature_importance['Absolute Coefficient'] = feature_importance['Coefficient'].abs()
    feature_importance = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)
    feature_importance.drop(columns=['Absolute Coefficient'], inplace=True)

    # Display the feature importance
    print(feature_importance)
    
    # Average detailed_results
    results_df = pd.DataFrame(detailed_results)
    results_df.drop(columns=['fit_time', 'score_time'], inplace=True)
    cols = results_df.columns
    average_values = results_df.mean(axis=0)
    

    # Convert the resulting Series to a DataFrame, reset index to make 'index' a column
    results_df = average_values.reset_index()

    # Rename columns to 'parameter' and 'value'
    results_df.columns = ['parameter', 'value']
    # results_df['parameter'] = cols
    
    
    return model_full, scaler, feature_importance, results_df

def predict_gel(gel_model, scaler, df_X: pd.DataFrame):
    
    # # Scale the new data using the previousy fitted scaler
    df_X = df_X.copy()
    
    # scaler = StandardScaler()
    df_X_scaled = scaler.transform(df_X)

    # Predict the probability of gelation for each condition
    pred_probs = gel_model.predict_proba(df_X_scaled)[:, 1]

    # Add probabilities to the DataFrame
    df_X['Gel'] = pred_probs
    df_X['Gel'] = df_X['Gel'].astype(float)
    
    return df_X

# def generate_scope()

def restrict_scope(df_X: pd.DataFrame, model, gel_prob_threshold: float = 0.5):
    """Restrict the scope of the experiment based on the predicted probability of gelation."""
    
    df_X = predict_gel(model, df_X)
    
    # Filter out conditions with a predicted probability of gelation above the threshold
    df_X = df_X[df_X['Gel'] <= gel_prob_threshold]
    df_X = df_X.drop(columns='Gel')
    df_X = df_X.reset_index(drop=True)
    
    return df_X

# Start with initial experiments
# Use the model to predict the probability of gelation for each condition
# Filter out conditions with a predicted probability of gelation above the threshold
# Generate new experiments based on the filtered conditions
# Repeat the process iteratively until the desired number of experiments is reached