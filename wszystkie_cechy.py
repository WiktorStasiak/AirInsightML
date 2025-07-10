import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, ConfusionMatrixDisplay

air_quality = pd.read_csv("updated_pollution_dataset.csv")

features = [
    "Temperature", "Humidity", "PM2.5", "PM10", "NO2", "SO2",
    "CO", "Proximity_to_Industrial_Areas", "Population_Density"
]
target = "Air Quality"

scaler = StandardScaler()
air_quality_scaled = air_quality.copy()
air_quality_scaled[features] = scaler.fit_transform(air_quality[features])

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(air_quality[feature], kde=True, color="blue")
    plt.title(f"Rozkład: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Liczność")
plt.tight_layout()
plt.show()

correlation_matrix = air_quality[features].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Macierz korelacji cech")
plt.show()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

k = 5
model_scores = []
X = air_quality_scaled[features]
y = air_quality_scaled[target]

for model_name, model in models.items():
    scores = cross_validate(model, X, y, cv=k,
                            scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])

    model_scores.append({
        "Model": model_name,
        "Accuracy": scores['test_accuracy'].mean(),
        "Precision": scores['test_precision_weighted'].mean(),
        "Recall": scores['test_recall_weighted'].mean(),
        "F1 Score": scores['test_f1_weighted'].mean()
    })

results = pd.DataFrame(model_scores)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
plt.figure(figsize=(16, 12))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    sns.barplot(data=results, x=metric, y="Model", palette="viridis")
    plt.title(f"Porównanie {metric} dla różnych modeli")
    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.legend(loc="best")
plt.tight_layout()
plt.show()

top_result = results.sort_values(by="F1 Score", ascending=False).iloc[0]
best_model_name = top_result["Model"]
best_model = models[best_model_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=best_model.classes_)
disp.plot(cmap='viridis')
plt.title(f"Macierz pomyłek: {best_model_name} (wszystkie cechy)")
plt.show()

print(f"Najlepszy model: {best_model_name} (wszystkie cechy)")
print(classification_report(y_test, y_pred))

metrics_df = pd.DataFrame([
    {"Metric": "Accuracy", "Value": accuracy_score(y_test, y_pred)},
    {"Metric": "Precision", "Value": precision_score(y_test, y_pred, average='weighted')},
    {"Metric": "Recall", "Value": recall_score(y_test, y_pred, average='weighted')},
    {"Metric": "F1 Score", "Value": f1_score(y_test, y_pred, average='weighted')}
])

print("\nWyniki najlepszego modelu:")
for metric in metrics:
    print(f"{metric}: {top_result[metric]:.4f}")

plt.figure(figsize=(8, 6))
sns.barplot(data=metrics_df, x="Value", y="Metric", palette="coolwarm")
plt.title("Porównanie metryk dla najlepszego modelu")
plt.xlabel("Wartość")
plt.ylabel("Metryka")
plt.tight_layout()
plt.show()