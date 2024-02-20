import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import pickle

def create_model(data): 
    # Split data into features and target variable
    X = data[['height(cm)', 'hemoglobin', 'Gtp', 'triglyceride', 'weight(kg)',
              'serum creatinine', 'age', 'HDL', 'ALT', 'LDL']]
    y = data['smoking']

    from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import pandas as pd

def create_model(data):
    # Split data into features and target variable
    X = data[['height(cm)', 'hemoglobin', 'Gtp', 'triglyceride', 'weight(kg)',
              'serum creatinine', 'age', 'HDL', 'ALT', 'LDL']]
    y = data['smoking']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define categorical and numerical columns
    categorical_columns = X.select_dtypes(include='category').columns.tolist()
    numerical_columns = X.select_dtypes(exclude='category').columns.tolist()

    # One-hot encode categorical columns if they exist
    if categorical_columns:
        X_train_categorical = pd.get_dummies(X_train[categorical_columns], drop_first=True)
        X_test_categorical = pd.get_dummies(X_test[categorical_columns], drop_first=True)
    else:
        X_train_categorical = pd.DataFrame()
        X_test_categorical = pd.DataFrame()

    # Scale numerical columns using RobustScaler
    scaler = RobustScaler()
    X_train_numerical = scaler.fit_transform(X_train[numerical_columns])
    X_test_numerical = scaler.transform(X_test[numerical_columns])

    # Merge one-hot encoded and scaled numerical features into one dataframe
    X_train_transformed = pd.concat([X_train_categorical, pd.DataFrame(X_train_numerical, columns=numerical_columns)], axis=1)
    X_test_transformed = pd.concat([X_test_categorical, pd.DataFrame(X_test_numerical, columns=numerical_columns)], axis=1)

    # Train XGBoost Model
    param_grid = {
        'n_estimators': [90],
        'max_depth': [5],
        'learning_rate': [0.18, 0.28],
        'subsample': [0.8, 0.9, 1.0],
        # Add more XGBoost-specific hyperparameters as needed
    }

    xgb_clf = XGBClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train_transformed, y_train)

    best_model = grid_search.best_estimator_

    y_pred_prob = best_model.predict_proba(X_test_transformed)[:, 1]
    y_pred = best_model.predict(X_test_transformed)

    validation_roc_auc = roc_auc_score(y_test, y_pred_prob)

    print("Best ROC AUC score found:", grid_search.best_score_)
    print("Validation ROC AUC of the best model:", validation_roc_auc)

    return best_model, scaler

def get_clean_data():
    data = pd.read_csv("data/train.csv")

    # Convert categorical columns to 'category' type
    data['dental caries'] = data['dental caries'].astype('category')
    data['smoking'] = data['smoking'].astype('category')
    data['hearing(left)'] = data['hearing(left)'].astype('category')
    data['hearing(right)'] = data['hearing(right)'].astype('category')
    data['Urine protein'] = data['Urine protein'].astype('category')

    return data

def main():
    data = get_clean_data()
    
    best_model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()