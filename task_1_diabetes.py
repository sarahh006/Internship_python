import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
dataset = pd.read_csv(data_url, names=column_names)
columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_to_fix:
    median_value = dataset[column].median()
    dataset[column] = dataset[column].replace(0, median_value)
X = dataset.drop(columns=['Outcome'])
y = dataset['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
diabetes_model = RandomForestClassifier(random_state=42)
diabetes_model.fit(X_train, y_train)
predictions = diabetes_model.predict(X_test)

overall_accuracy = accuracy_score(y_test, predictions)
print("--- Model Evaluation ---")
print("Overall Accuracy:", round(overall_accuracy * 100, 2), "%")
print("---------------------------------")
print("Detailed Performance Report:")
print(classification_report(y_test, predictions))