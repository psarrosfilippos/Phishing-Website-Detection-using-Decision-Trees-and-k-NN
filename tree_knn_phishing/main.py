import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

# Load the dataset
data, meta = arff.loadarff('Training Dataset.arff')
data = pd.DataFrame(data)

# Encode 'Result' label from byte to int
label_encoder = LabelEncoder()
data['Result'] = label_encoder.fit_transform(data['Result'])

# Split features and labels
X = data.drop('Result', axis=1)
y = data['Result']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------- Decision Tree Optimization ----------
dt_leaf_values = [2, 51, 101, 151, 201, 251, 301]
dt_accuracies = []
dt_recalls = []
dt_f1_scores = []

for max_leaf_nodes in dt_leaf_values:
    dt_model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=1)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    dt_accuracies.append(accuracy_score(y_test, y_pred_dt))
    dt_recalls.append(recall_score(y_test, y_pred_dt))
    dt_f1_scores.append(f1_score(y_test, y_pred_dt))

# Best DT model
best_index_dt = np.argmax(dt_accuracies)
best_leaf_nodes = dt_leaf_values[best_index_dt]
best_accuracy_dt = dt_accuracies[best_index_dt]

dt_model = DecisionTreeClassifier(max_leaf_nodes=best_leaf_nodes, random_state=1)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# ---------- k-NN Optimization ----------
k_values = list(range(1, 31))
knn_accuracies = []
knn_recalls = []
knn_f1_scores = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    knn_accuracies.append(accuracy_score(y_test, y_pred_knn))
    knn_recalls.append(recall_score(y_test, y_pred_knn))
    knn_f1_scores.append(f1_score(y_test, y_pred_knn))

# Best k-NN model
best_index_knn = np.argmax(knn_accuracies)
best_k = k_values[best_index_knn]
best_accuracy_knn = knn_accuracies[best_index_knn]

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# ---------- Metrics ----------
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

# ---------- Confusion Matrices ----------
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# Percentages
legit_percent_dt = conf_matrix_dt[0, 0] / np.sum(conf_matrix_dt) * 100
phishing_percent_dt = conf_matrix_dt[1, 1] / np.sum(conf_matrix_dt) * 100

legit_percent_knn = conf_matrix_knn[0, 0] / np.sum(conf_matrix_knn) * 100
phishing_percent_knn = conf_matrix_knn[1, 1] / np.sum(conf_matrix_knn) * 100

# ---------- Visualizations ----------

# Confusion matrix - Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion matrix - k-NN
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - k-NN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Bar Chart: Legit & Phishing Detection
categories = ['Decision Tree', 'k-NN']
legit_percentages = [legit_percent_dt, legit_percent_knn]
phishing_percentages = [phishing_percent_dt, phishing_percent_knn]

x = np.arange(len(categories))
width = 0.4

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, legit_percentages, width, label='Legit', color='blue')
plt.bar(x + width/2, phishing_percentages, width, label='Phishing', color='green')
plt.xticks(x, categories)
plt.ylabel("Detection Rate (%)")
plt.title("Legit vs Phishing Detection")
plt.legend()
plt.show()

# Metrics Summary Table
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Recall", "F1 Score"],
    "Decision Tree": [best_accuracy_dt, recall_dt, f1_dt],
    "k-NN": [best_accuracy_knn, recall_knn, f1_knn]
})

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results_df.values,
                 colLabels=results_df.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
plt.title("Performance Summary")
plt.show()

# Bar Chart: Precision, Recall, F1 Comparison
metrics = ['Precision', 'Recall', 'F1 Score']
dt_metrics = [precision_dt, recall_dt, f1_dt]
knn_metrics = [precision_knn, recall_knn, f1_knn]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, dt_metrics, width, label='Decision Tree', color='blue')
plt.bar(x + width/2, knn_metrics, width, label='k-NN', color='green')
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Precision, Recall, F1 Comparison")
plt.legend()
plt.show()

# Line Plot: Decision Tree Accuracy by max_leaf_nodes
plt.figure(figsize=(10, 6))
plt.plot(dt_leaf_values, dt_accuracies, marker='o', color='blue')
plt.xlabel('max_leaf_nodes')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs max_leaf_nodes')
plt.grid(True)
plt.show()

# Line Plot: k-NN Accuracy by k
plt.figure(figsize=(10, 6))
plt.plot(k_values, knn_accuracies, marker='o', color='green')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('k-NN Accuracy vs k')
plt.grid(True)
plt.show()
