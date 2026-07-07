import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

local_data = {
    'PassengerID': [1, 2, 3, 4, 5, 6, 7, 8],
    'TicketClass': [3, 1, 3, 1, 3, 3, 1, 3],
    'Gender': ['Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Male'],
    'Age': [22.0, 38.0, 26.0, 35.0, 35.0, np.nan, 54.0, 2.0],
    'Siblings': [1, 1, 0, 1, 0, 0, 0, 3],
    'Parents': [0, 0, 0, 0, 0, 0, 0, 1],
    'TicketFare': [7.25, 71.28, 7.92, 53.10, 8.05, 8.45, 51.86, 21.07],
    'BoardingPort': ['Southampton', 'Cherbourg', 'Southampton', 'Southampton', 'Southampton', np.nan, 'Southampton', 'Southampton'], 
    'Survived': [0, 1, 1, 1, 0, 0, 0, 0]
}
raw_titanic = pd.DataFrame(local_data)
median_age_value = raw_titanic['Age'].median()
raw_titanic['Age'] = raw_titanic['Age'].fillna(median_age_value)
most_common_port = raw_titanic['BoardingPort'].mode()[0]
raw_titanic['BoardingPort'] = raw_titanic['BoardingPort'].fillna(most_common_port)
raw_titanic = raw_titanic.drop(columns=['PassengerID'])
raw_titanic['TotalFamilySize'] = raw_titanic['Siblings'] + raw_titanic['Parents'] + 1
raw_titanic['Gender'] = raw_titanic['Gender'].replace({'Male': 0, 'Female': 1})
clean_titanic = pd.get_dummies(raw_titanic, columns=['BoardingPort'], drop_first=True)

X_titanic = clean_titanic.drop(columns=['Survived'])
y_titanic = clean_titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(X_titanic, y_titanic, test_size=0.2, random_state=42)

data_scaler = StandardScaler()
X_train_scaled = data_scaler.fit_transform(X_train)
X_test_scaled = data_scaler.transform(X_test)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)
logistic_predictions = logistic_model.predict(X_test_scaled)

tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

log_accuracy = accuracy_score(y_test, logistic_predictions)
tree_accuracy = accuracy_score(y_test, tree_predictions)
print("Logistic Regression Model Accuracy: ", round(log_accuracy * 100, 2), "%")
print("Decision Tree Model Accuracy:        ", round(tree_accuracy * 100, 2), "%")