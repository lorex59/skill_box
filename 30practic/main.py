import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def filter_data(df):
   columns_to_drop = [
       'id',
       'url',
       'region',
       'region_url',
       'price',
       'manufacturer',
       'image_url',
       'description',
       'posting_date',
       'lat',
       'long'
   ]
   # Возвращаем копию датафрейма, inplace тут делать нельзя!
   return df.drop(columns_to_drop, axis=1)

def calculate_outliers(data):
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    iqr = q75 - q25
    return (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

def emissions_data(df):
    df_new = df.copy()
    boundaries = calculate_outliers(df_new['year'])
    df_new.loc[df_new['year'] > boundaries[1], 'year'] = round(boundaries[1])
    df_new.loc[df_new['year'] < boundaries[0], 'year'] = round(boundaries[0])
    return df_new


def short_model(x):
    if not pd.isna(x):
        return x.lower().split(' ')[0]
    else:
        return x


def new_features(df):
    df_new = df.copy()
    
    df_new.loc[:, 'short_model'] = df_new['model'].apply(short_model)
    df_new.loc[:, 'age_category'] =  df_new['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df_new
    
def change(x):
    if x == 'high':
        return 2
    elif x == 'medium':
        return 1
    else:
        return 0
    
def main():
    print("Loan Prediction Pipeline")
    
    path = r'D:\STUDY\skill_box\30practic\30.6 homework.csv'
    df = pd.read_csv(path)
    X = df.drop(['price_category'], axis=1)
    y = df['price_category'].apply(change)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaller', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    filter = Pipeline(steps=[
        ('delete', FunctionTransformer(filter_data)),
        ('emissions', FunctionTransformer(emissions_data)),
        ('new_features', FunctionTransformer(new_features)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ], remainder='passthrough')
    
    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    )
    
    best_score = 0.0
    best_pipe = None
    
    for model in models:
        pipe = Pipeline(steps=[
            ('filter', filter),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        
        print(f"model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}")
    
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'homework.pkl')
    
if __name__ == '__main__':
    main()
