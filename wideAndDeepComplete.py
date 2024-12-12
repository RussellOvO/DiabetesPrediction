import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns  # Added for better visualization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # For confusion matrix
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
import pandas as pd
import numpy as np

# Load data
file_path = '/home/wsoxl/503/Project/diabetes_data.csv'
data = pd.read_csv(file_path)

# Separate features and target
target = 'Diagnosis'
X = data.drop(columns=['PatientID', 'DoctorInCharge', target])
y = data[target]

# Encode target if it's not already binary
# Assuming 'Diagnosis' is categorical (e.g., 'Positive', 'Negative'), otherwise skip
if y.dtype == 'object' or y.dtype.name == 'category':
    # Update the mapping based on your dataset's actual labels
    # Example:
    y = y.map({'Positive': 1, 'Negative': 0})  # Adjust if labels are different

# Identify categorical and numerical features
categorical_features = ['Gender', 'Ethnicity', 'SocioeconomicStatus', 'EducationLevel', 'Smoking', 
                        'FamilyHistoryDiabetes', 'Hypertension', 'GestationalDiabetes',
                        'PolycysticOvarySyndrome', 'PreviousPreDiabetes', 'AntihypertensiveMedications',
                        'Statins', 'AntidiabeticMedications', 'FrequentUrination', 'ExcessiveThirst', 
                        'UnexplainedWeightLoss', 'BlurredVision', 'SlowHealingSores', 'TinglingHandsFeet', 
                        'HeavyMetalsExposure', 'OccupationalExposureChemicals', 'WaterQuality'
                       ]
numerical_features = [col for col in X.columns if col not in categorical_features]

# Train-test split with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define Wide and Deep model
class WideAndDeepModel(Model):
    def __init__(self, wide_dim, deep_input_dim, deep_hidden_units, output_dim):
        super(WideAndDeepModel, self).__init__()
        
        # Wide component
        self.wide = layers.Dense(
            units=output_dim, 
            activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)
        )

        # Deep component
        self.deep = tf.keras.Sequential([
            layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            ) for units in deep_hidden_units
        ] + [
            layers.Dense(
                units=output_dim, 
                activation='linear',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
            )
        ])
    
    def call(self, inputs):
        wide_input, deep_input = inputs
        wide_output = self.wide(wide_input)
        deep_output = self.deep(deep_input)
        combined = wide_output + deep_output
        return tf.nn.sigmoid(combined)

# Define dimensions
wide_dim = X_train_processed.shape[1]
deep_input_dim = len(numerical_features)

# Instantiate the model
model = WideAndDeepModel(
    wide_dim=wide_dim,
    deep_input_dim=deep_input_dim,
    deep_hidden_units=[64, 32],
    output_dim=1
)

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Start timing
start_time = time.time()

# Train the model
history = model.fit(
    [X_train_processed, X_train[numerical_features].values], y_train,
    validation_data=([X_test_processed, X_test[numerical_features].values], y_test),
    epochs=100,
    batch_size=32,
    verbose=1  # Set to 1 to see training progress
)

# End timing
end_time = time.time()
total_training_time = end_time - start_time
print(f"Total Training Time: {total_training_time:.2f} seconds")

# Evaluate the model
loss, accuracy = model.evaluate([X_test_processed, X_test[numerical_features].values], y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Predict probabilities
y_test_pred_probs = model.predict([X_test_processed, X_test[numerical_features].values])

# Threshold probabilities to get binary predictions
threshold = 0.5
y_test_pred_binary = (y_test_pred_probs > threshold).astype(int).flatten()

# Calculate F1-score
f1 = f1_score(y_test, y_test_pred_binary)
print(f"F1-Score: {f1:.4f}")

# Print classification report for additional metrics
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_binary))

# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred_binary)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])

# Plot Confusion Matrix using sklearn's ConfusionMatrixDisplay
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Alternatively, plot Confusion Matrix using seaborn for enhanced visualization
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Plot the Accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot the Loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
