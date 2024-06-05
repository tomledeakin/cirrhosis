import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import pickle
from collections import Counter
from imblearn.pipeline import Pipeline as ImbPipeline

# Load data
train_data = pd.read_csv('cirrhosis_train.csv')
test_data = pd.read_csv('cirrhosis_test.csv')

target = 'Status'
y = train_data[target]
X = train_data.drop(columns=[target, 'trainID'], axis=1)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=42, test_size=0.3, stratify=y_encoded)

# Nominal data
nominal_columns = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']

# Ordinal data
ordinal_columns = ['Stage']

# Numerical data
numerical_columns = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
                     'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']

preprocessor = ColumnTransformer(transformers=[
    ('ord', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ]), ordinal_columns),
    ('nom', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder())
    ]), nominal_columns),
    ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_columns)
])

# Oversampling strategies
smote = SMOTE(random_state=42)
ros = RandomOverSampler(random_state=42)

# Define the pipeline
cls = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('ros', ros),
    ('smote', smote),
    ('model', RandomForestClassifier(random_state=2024))
])

# Parameter grid for RandomizedSearchCV
param_grid = {
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    'preprocessor__nom__imputer__strategy': ['most_frequent', 'constant'],
    'preprocessor__ord__imputer__strategy': ['most_frequent', 'median'],
    'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
    'smote__k_neighbors': [2, 3],
    'ros__sampling_strategy': [{1: 22}, {1: 28}, {1: 30}],
    'smote__sampling_strategy': ['auto', {1: 35}, {1: 45}, {1: 55}],
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__max_features': ['sqrt', 'log2'],
    'model__criterion': ['gini', 'entropy'],
}

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = RandomizedSearchCV(cls, param_distributions=param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=1, n_iter=500)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
best_model = grid_search.best_estimator_

filename = 'best_model.pkl'
# pickle.dump(grid_search, open(filename, 'wb'))
model = pickle.load(open(filename, 'rb'))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0.0))

# Predict on test data
testID = test_data['testID']
test_data = test_data.drop(columns=['Status', 'testID'], axis=1)
y_pred_test = model.predict(test_data)

# Inverse transform the predictions to get the original labels
y_pred_test = label_encoder.inverse_transform(y_pred_test)

# Create a DataFrame with testID and predicted Status
prediction_df = pd.DataFrame({
    'testID': testID,
    'Status': y_pred_test
})

# Save the DataFrame to a CSV file
prediction_file_path = 'cirrhosis_predicted.csv'
prediction_df.to_csv(prediction_file_path, index=False)

#               precision    recall  f1-score   support
#
#            0       0.86      0.86      0.86        36
#            1       1.00      0.25      0.40         4
#            2       0.81      0.89      0.85        28
#
#     accuracy                           0.84        68
#    macro avg       0.89      0.67      0.70        68
# weighted avg       0.85      0.84      0.83        68
