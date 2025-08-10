import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from utils import extract_features

# Define paths
DATASET_PATH = 'dataset/'
MODEL_SAVE_DIR = 'saved_models/'
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'fruit_ripeness_svm_model.pkl')
SCALER_PATH = os.path.join(MODEL_SAVE_DIR, 'feature_scaler.pkl')

# Ensure the save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 1. Load Data and Extract Features
print("Loading data and extracting features...")
features = []
labels = []

if not os.listdir(DATASET_PATH):
    print(f"Error: The 'dataset' directory is empty. Please add image folders to it.")
    exit()

for class_label in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_label)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            feature_vector = extract_features(image_path)
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(class_label)

print(f"Extracted features for {len(features)} images.")

if not features:
    print("Error: No features were extracted. Check your images and dataset path.")
    exit()

# 2. Split Data and Scale Features
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Hyperparameter Tuning with GridSearchCV
print("Starting Grid Search for best hyperparameters...")
param_grid = {
    'C': [1, 10, 100, 1000],
    'gamma': [0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2, cv=3, n_jobs=-1) # cv=3 for faster training on small datasets
grid.fit(X_train_scaled, y_train)

best_svm_model = grid.best_estimator_
print("\nBest parameters found:", grid.best_params_)

# 4. Evaluate the Best Model
print("\nEvaluating the best model on the test set...")
y_pred = best_svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Save the Model and Scaler
print("\nSaving the model and scaler...")
joblib.dump(best_svm_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("Training complete. Model and scaler saved successfully.")