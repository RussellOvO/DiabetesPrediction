import pandas as pd
import lightgbm as lgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import time

# Initialize dictionaries to store CCR results for all datasets
all_train_ccr = {}
all_valid_ccr = {}

# Define datasets, drop columns, and their target columns
datasets_info = [
    {
        'path': '/home/wsoxl/503/Project/diabetes_2000.csv',
        'drop': [],  # No additional features to drop
        'target': 'Outcome'
    },
    {
        'path': '/home/wsoxl/503/Project/diabetes_data.csv',
        'drop': ['PatientID', 'DoctorInCharge'],  # Drop additional features
        'target': 'Diagnosis'
    },
    {
        'path': '/home/wsoxl/503/Project/diabetes_pima_indian.csv',
        'drop': [],  # No additional features to drop
        'target': 'Outcome'
    }
]

# Function to train and evaluate a LightGBM model
def train_and_evaluate_model(dataset_path, drop_features, target_column, params, model_index):
    print(f"\nProcessing Dataset {model_index}...")

    # Measure start time
    start_time = time.time()
    
    # Load dataset
    data = pd.read_csv(dataset_path)
    
    # Separate features (X) and target (y)
    y = data[target_column]
    X = data.drop(columns=[target_column] + drop_features)  # Drop target column and additional columns

    # Handle low-variance features
    vt = VarianceThreshold(threshold=0.01)  # Threshold for minimum variance
    X = vt.fit_transform(X)
    selected_features = [data.drop(columns=[target_column] + drop_features).columns[i] for i, selected in enumerate(vt.get_support()) if selected]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(data=X_train, label=y_train, feature_name=selected_features)
    test_data = lgb.Dataset(data=X_test, label=y_test, feature_name=selected_features, reference=train_data)
    
    # Initialize a dictionary to manually track CCR results
    eval_results = {'train_ccr': [], 'valid_ccr': []}
    
    # Define a custom callback to compute and store CCR values
    def record_ccr_callback(env):
        y_train_pred = (env.model.predict(X_train) > 0.5).astype(int)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        y_valid_pred = (env.model.predict(X_test) > 0.5).astype(int)
        valid_accuracy = accuracy_score(y_test, y_valid_pred)
        
        eval_results['train_ccr'].append(train_accuracy)
        eval_results['valid_ccr'].append(valid_accuracy)
    
    # Train the LightGBM model
    print(f"Training LightGBM model for Dataset {model_index}...")
    lgb_model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=10),
            record_ccr_callback
        ]
    )
    
    # Measure end time and calculate runtime
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Dataset {model_index} - Training Runtime: {runtime:.2f} seconds")
    
    # Store CCR results for plotting
    all_train_ccr[f"Dataset {model_index}"] = eval_results['train_ccr']
    all_valid_ccr[f"Dataset {model_index}"] = eval_results['valid_ccr']
    
    # Evaluate the model
    y_pred_prob = lgb_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    
    print(f"Dataset {model_index} - Accuracy: {accuracy * 100:.2f}%, ROC-AUC: {roc_auc:.2f}, F1-Score: {f1:.2f}")
    print(f"Dataset {model_index} - Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot feature importance
    print(f"Plotting feature importance for Dataset {model_index}...")
    lgb.plot_importance(lgb_model, max_num_features=10, importance_type='gain')
    plt.title(f"Feature Importance for Dataset {model_index}")
    plt.show()

    # Return runtime and F1-score
    return runtime, f1

# LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 50,
    'max_depth': 8,
    'min_data_in_leaf': 10,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0.0,
    'verbose': -1
}

# Train and evaluate models for each dataset
for index, dataset in enumerate(datasets_info, start=1):
    runtime, f1 = train_and_evaluate_model(dataset['path'], dataset['drop'], dataset['target'], params, index)
    print(f"Dataset {index} - Training Runtime: {runtime:.2f} seconds, F1-Score: {f1:.2f}")

# Plot Training CCR for all datasets
plt.figure(figsize=(12, 6))
for dataset, train_ccr in all_train_ccr.items():
    plt.plot(train_ccr, label=f"{dataset} Training CCR")

plt.xlabel('Boosting Rounds')
plt.ylabel('CCR (Correct Classification Rate)')
plt.title('Training CCR Trends for All Datasets')
plt.legend()
plt.grid()
plt.show()

# Plot Validation CCR for all datasets
plt.figure(figsize=(12, 6))
for dataset, valid_ccr in all_valid_ccr.items():
    plt.plot(valid_ccr, label=f"{dataset} Validation CCR")

plt.xlabel('Boosting Rounds')
plt.ylabel('CCR (Correct Classification Rate)')
plt.title('Validation CCR Trends for All Datasets')
plt.legend()
plt.grid()
plt.show()
