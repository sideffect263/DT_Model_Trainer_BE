import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import numpy as np
import json
import joblib
import os


def train_model(features, target, model_type, session_id):
    print("Training request received")
    print("Features:", features)
    print("Target:", target)
    print("Model Type:", model_type)
    random_seed = 123
    np.random.seed(random_seed)
    
    try:
        # Load the dataset
        data = pd.read_csv(f'uploads/{session_id}/temp.csv')
        print("Data loaded successfully")

        # Define features (X) and target (y)
        X = data[features]
        y = data[target]

        # Split the data into training (80%) and testing (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
        print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
        
        # Ensure no data leakage
        print(f"Training target values sample: {y_train[:5].values}")
        print(f"Testing target values sample: {y_test[:5].values}")

        # Train the model based on the model type
        if model_type == 'Regression':
            model = DecisionTreeRegressor(random_state=8123)
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 10, 20]
            }
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Model trained successfully. MSE: {mse}")
            print("Sample predictions:")
            for actual, predicted in zip(y_test[:5], y_pred[:5]):
                print(f"Actual: {actual}, Predicted: {predicted}")
            print(f"Best Parameters: {grid_search.best_params_}")
            feature_importances = best_model.feature_importances_
            print("Feature Importances:")
            for feature, importance in zip(features, feature_importances):
                print(f"{feature}: {importance}")
            
            # Save the model to a file
            model_path = f'uploads/{session_id}/trained_model.pkl'
            joblib.dump(best_model, model_path)
            results = {'mse': mse, 'best_params': grid_search.best_params_, 'feature_importances': feature_importances.tolist()}
        
        elif model_type == 'Classification':
            model = DecisionTreeClassifier(random_state=8123)
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 10, 20]
            }
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            print(f"Model trained successfully. Accuracy: {accuracy}")
            print(f"Best Parameters: {grid_search.best_params_}")
            feature_importances = best_model.feature_importances_
            print("Feature Importances:")
            for feature, importance in zip(features, feature_importances):
                print(f"{feature}: {importance}")
            
            # Save the model to a file
            model_path = f'uploads/{session_id}/trained_model.pkl'
            joblib.dump(best_model, model_path)
            results = {'accuracy': accuracy, 'report': report, 'best_params': grid_search.best_params_, 'feature_importances': feature_importances.tolist()}
        
                # Print results as JSON string
        print("Results:")
        print(json.dumps(results))
        
    except Exception as e:
        print("An error occurred:", str(e))
        results = {'error': str(e)}
        print(json.dumps(results))

if __name__ == "__main__":
    features = sys.argv[1].split(',')
    target = sys.argv[2]
    model_type = sys.argv[3]
    session_id = sys.argv[4]

    train_model(features, target, model_type, session_id)

