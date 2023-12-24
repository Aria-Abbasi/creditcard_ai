import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import silhouette_score

# Load dataset
dataset = pd.read_csv('creditcard.csv')

# Data preprocessing
sc = StandardScaler()
dataset['normalizedAmount'] = sc.fit_transform(dataset['Amount'].values.reshape(-1, 1))
dataset = dataset.drop(['Amount', 'Time'], axis=1)
X = dataset.iloc[:, dataset.columns != 'Class']
y = dataset.iloc[:, dataset.columns == 'Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# K-Means Clustering
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_train)
X_train['Cluster'] = y_kmeans

# Clustering Evaluation
silhouette_avg = silhouette_score(X_train.drop('Cluster', axis=1), y_kmeans)
print("Silhouette Score for Clustering:", silhouette_avg)

# Logistic Regression Classification
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train.drop('Cluster', axis=1), y_train.values.ravel())
y_pred = classifier.predict(X_test)

# Classification Evaluation
classification_metrics = precision_recall_fscore_support(y_test, y_pred, average='binary')
print("Classification Metrics (Precision, Recall, F1-Score):", classification_metrics)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

