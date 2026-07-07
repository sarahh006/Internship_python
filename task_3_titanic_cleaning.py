import pandas as pd
import numpy as np
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

print("-------Original Raw Data--------")
print(raw_titanic)
print("--------------------------------------------")

median_age_value = raw_titanic['Age'].median()
raw_titanic['Age'] = raw_titanic['Age'].fillna(median_age_value)
most_common_port = raw_titanic['BoardingPort'].mode()[0]
raw_titanic['BoardingPort'] = raw_titanic['BoardingPort'].fillna(most_common_port)
raw_titanic = raw_titanic.drop(columns=['PassengerID'])
raw_titanic['TotalFamilySize'] = raw_titanic['Siblings'] + raw_titanic['Parents'] + 1
raw_titanic['Gender'] = raw_titanic['Gender'].replace({'Male': 0, 'Female': 1})
clean_titanic = pd.get_dummies(raw_titanic, columns=['BoardingPort'], drop_first=True)
print(clean_titanic)