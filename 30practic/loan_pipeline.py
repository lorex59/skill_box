import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import os

def main():
    print("Loan Prediction Pipeline")
    path = r'D:\STUDY\skill_box\30practic\data\loan_train.csv'
    df = pd.read_csv(path).drop("Loan_ID", axis=1)
    x = df.drop("Loan_Status", axis=1)
    y = df['Loan_Status'].apply(lambda x: 1.0 if x == "Y" else 0.0)
    
    numerical_features = x.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = x.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])
    
    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64))
    )
    
    best_score = .0
    best_pipe = None
    
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, x, y, cv= 4, scoring='accuracy')
        print(f"model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}")
    
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'loan_pipeline.pkl')
    
        
if __name__ == '__main__':
    main()
