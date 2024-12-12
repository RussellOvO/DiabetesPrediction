import pandas as pd
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #忽略独立显卡
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

# Step 1: Load the dataset
file_path = "diabetes.csv"  # 请确保文件路径正确
diabetes_data = pd.read_csv(file_path)

# Step 2: Select relevant features and target
selected_features = [
    # "BMI", "Hypertension", "FastingBloodSugar", "HbA1c", 
    # "CholesterolTotal", "CholesterolLDL", "CholesterolHDL"

    # "PatientID", "Age", "Gender", "Ethnicity", "SocioeconomicStatus", 
    # "EducationLevel", "BMI", "Smoking", "AlcoholConsumption", "PhysicalActivity",
    # "DietQuality", "SleepQuality", "FamilyHistoryDiabetes", "GestationalDiabetes", 
    # "PolycysticOvarySyndrome", "PreviousPreDiabetes", "Hypertension", "SystolicBP", 
    # "DiastolicBP", "FastingBloodSugar", "HbA1c", "SerumCreatinine", "BUNLevels", 
    # "CholesterolTotal", "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
    # "AntihypertensiveMedications", "Statins", "AntidiabeticMedications", "FrequentUrination",
    # "ExcessiveThirst", "UnexplainedWeightLoss", "FatigueLevels", "BlurredVision", 
    # "SlowHealingSores", "TinglingHandsFeet", "QualityOfLifeScore", "HeavyMetalsExposure", 
    # "OccupationalExposureChemicals", "WaterQuality", "MedicalCheckupsFrequency", 
    # "MedicationAdherence", "HealthLiteracy", "DoctorInCharge"

    # "FastingBloodSugar", "HbA1c"
    # "BMI", "Glucose"
]
target_column = "Diagnosis"
target_column = "Outcome"

# Step 3: Prepare features and labels
X = diabetes_data[selected_features].values
y = diabetes_data[target_column].values
# # 检查非数值列
# non_numeric_columns = diabetes_data.select_dtypes(include=['object']).columns   
# diabetes_data = pd.get_dummies(diabetes_data, columns=non_numeric_columns, drop_first=True)
# X = diabetes_data.drop(columns=["Diagnosis"]).values  # 使用所有特征
# y = diabetes_data["Diagnosis"].values 

non_numeric_columns = diabetes_data.select_dtypes(include=['object']).columns   
diabetes_data = pd.get_dummies(diabetes_data, columns=non_numeric_columns, drop_first=True)
X = diabetes_data.drop(columns=["Outcome"]).values  # 使用所有特征
y = diabetes_data["Outcome"].values 

# Step 4: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Build the ANN model
model = Sequential()
model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))  # Input layer
# model.add(Dropout(0.5))
model.add(Dense(16, activation='relu')) 
# model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))                              # Hidden layer
model.add(Dense(64, activation='relu'))  # 第二隐藏层
model.add(Dense(16, activation='relu'))  # 第三隐藏层
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))                           # Output layer (binary classification)

# Step 7: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])


# 训练并验证模型
start_train_time = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500,
    batch_size=16,
    verbose=1
)
end_train_time = time.time()

# 评估模型
start_predict_time = time.time() 
y_pred_prob = model.predict(X_test).flatten()  # Predict probabilities
y_pred = (y_pred_prob > 0.5).astype(int)       # Convert to binary predictions
end_predict_time = time.time()

# Plot CCR curves
# plt.figure(figsize=(10, 6))
# Training CCR curve
plt.plot(history.history['accuracy'], label='Train CCR')
# Testing CCR curve
plt.plot(history.history['val_accuracy'], label='Test CCR')
# Add labels and legend
plt.title('CCR (Correct Classification Rate) Curves')
plt.xlabel('Epochs')
plt.ylabel('CCR (Accuracy)')
plt.legend()
plt.grid(True)
plt.show()

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Print confusion matrix
conf_matrix_text = f"""
Confusion Matrix:
[[{conf_matrix[0, 0]}  {conf_matrix[0, 1]}]
 [{conf_matrix[1, 0]}  {conf_matrix[1, 1]}]]
"""
print(conf_matrix_text)

# Print training and prediction time
train_time = end_train_time - start_train_time
predict_time = end_predict_time - start_predict_time

print(f"Training Time: {train_time:.2f} seconds")
print(f"Prediction Time: {predict_time:.2f} seconds")