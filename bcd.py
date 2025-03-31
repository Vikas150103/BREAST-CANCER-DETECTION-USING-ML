import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'data_breast cancer.csv'  # Ensure this points to the correct CSV file
data = pd.read_csv(file_path)

# Data preprocessing
# Assuming 'diagnosis' is the target variable, encode it to 1 (malignant) and 0 (benign)
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Drop any unnecessary columns like 'id' or unnamed columns
data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Define features (X) and target (y)
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(probability=True, random_state=42)
knn_model = KNeighborsClassifier()
gradient_boosting_model = GradientBoostingClassifier(random_state=42)

# Create a dictionary to store the models and their names
models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
    "SVM": svm_model,
    "KNN": knn_model,
    "Gradient Boosting": gradient_boosting_model
}

# Initialize lists to store the evaluation metrics
model_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Train each model and evaluate its performance using cross-validation and testing
for name, model in models.items():
    # Cross-validation for accuracy
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} - Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_preds = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)

    # Append the results
    model_names.append(name)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Create a DataFrame to store all the results
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1-Score': f1_scores
})

# Display the results
print(results_df)

# Plotting the comparison graph for Accuracy with color gradients
plt.figure(figsize=(12, 8))

# Custom color palette with gradient
custom_palette = sns.light_palette("green", as_cmap=False, n_colors=5)

sns.barplot(x='Model', y='Accuracy', data=results_df, palette=custom_palette)
plt.title('Model Comparison - Accuracy', fontsize=16)
plt.ylabel('Accuracy Score', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Precision comparison graph with another color gradient
plt.figure(figsize=(12, 8))
custom_palette2 = sns.light_palette("blue", as_cmap=False, n_colors=5)
sns.barplot(x='Model', y='Precision', data=results_df, palette=custom_palette2)
plt.title('Model Comparison - Precision', fontsize=16)
plt.ylabel('Precision Score', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Recall comparison graph with another color gradient
plt.figure(figsize=(12, 8))
custom_palette3 = sns.light_palette("orange", as_cmap=False, n_colors=5)
sns.barplot(x='Model', y='Recall', data=results_df, palette=custom_palette3)
plt.title('Model Comparison - Recall', fontsize=16)
plt.ylabel('Recall Score', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Plotting the ROC curves
plt.figure(figsize=(10, 8))

for name, model in models.items():
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)

    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.show()

# Feature Importance for Random Forest
importances = random_forest_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance for Random Forest
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Random Forest Feature Importance')
plt.show()

# Hyperparameter Tuning Example for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best Parameters for Random Forest: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Plot the feature importance for Random Forest with custom color palette
plt.figure(figsize=(10, 6))

# Custom color palette with different colors and shades
colors = sns.color_palette("viridis", len(feature_importance_df))

# Plot the barplot with the custom colors
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette=colors)

plt.title('Random Forest Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()
