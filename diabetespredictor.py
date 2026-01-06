import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
diabetes_dataset = pd.read_csv("diabetes.csv")
print("\n--- Data Preview (First 5 Rows):")
print(diabetes_dataset.head())
print("\n--- Outcome Values:")
print(diabetes_dataset['Outcome'])
print("\n--- Standardizing Features:")
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print("\nStandardized Feature Data (sample):")
print(standardized_data[:5])
X = standardized_data
Y = diabetes_dataset['Outcome']
print(f"\n--- Shape of data arrays:")
print(f"Total dataset shape: {X.shape}")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("\n--- Accuracy Scores:")
print(f"Training data accuracy : {training_data_accuracy}")
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print(f"Test data accuracy : {test_data_accuracy}")
input_data = (3, 130, 70, 25, 100, 28.5, 0.450, 40)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
print("\n--- Scaled Input Data for Prediction:")
print(std_data)
prediction = classifier.predict(std_data)
print("\n--- Prediction:")
print(prediction)
print("\n--- Diagnosis Result:")
if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")