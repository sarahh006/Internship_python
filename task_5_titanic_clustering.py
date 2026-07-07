from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
X_unsupervised = clean_titanic.drop(columns=['Survived'])

unsupervised_scaler = StandardScaler()
X_scaled_matrix = unsupervised_scaler.fit_transform(X_unsupervised)
pca_reducer = PCA(n_components=2)
reduced_2d_data = pca_reducer.fit_transform(X_scaled_matrix)
kmeans_clusterer = KMeans(n_clusters=2, random_state=42, n_init=10)
assigned_clusters = kmeans_clusterer.fit_predict(X_scaled_matrix)

plt.figure(figsize=(8, 6))
x_coordinates = reduced_2d_data[:, 0]
y_coordinates = reduced_2d_data[:, 1]
plt.scatter(x_coordinates, y_coordinates, c=assigned_clusters, cmap='bwr', alpha=0.6)
plt.title('Unsupervised Clustering on Titanic Passenger Characteristics')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()