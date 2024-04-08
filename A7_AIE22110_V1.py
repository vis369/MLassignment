import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

# Function to train and test MLPClassifier
def train_and_test_mlp(X_train, y_train, X_test, y_test):
    mlp_classifier = MLPClassifier(learning_rate_init=0.05, momentum=0.9)
    mlp_classifier.fit(X_train, y_train)
    predictions = mlp_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Function to load data
def load_data(excel_path):
    df = pd.read_excel(excel_path)
    features = df.drop('Disease', axis=1).to_numpy()
    labels = df['Disease'].to_numpy()
    return features, labels

# Path to the Excel file containing data
excel_path = "C:/Users/Varun/Downloads/extracted_features_1.xlsx"

# Load data
X, y = load_data(excel_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform hyperparameter tuning for Perceptron
param_grid_perceptron = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'max_iter': [1000, 2000, 3000],
    'tol': [1e-3, 1e-4, 1e-5]
}
perceptron = Perceptron()
perceptron_random = RandomizedSearchCV(estimator=perceptron, param_distributions=param_grid_perceptron, n_iter=100, cv=5, random_state=42, n_jobs=-1)
perceptron_random.fit(X_train, y_train)

# Perform hyperparameter tuning for MLPClassifier
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (200,)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'learning_rate_init': [0.01, 0.05, 0.1],
    'max_iter': [200, 400, 600],
    'tol': [1e-3, 1e-4, 1e-5]
}
mlp = MLPClassifier()
mlp_random = RandomizedSearchCV(estimator=mlp, param_distributions=param_grid_mlp, n_iter=100, cv=5, random_state=42, n_jobs=-1)
mlp_random.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters for Perceptron:", perceptron_random.best_params_)
print("Best parameters for MLPClassifier:", mlp_random.best_params_)

# Evaluate the models using the best parameters
perceptron_best = perceptron_random.best_estimator_
mlp_best = mlp_random.best_estimator_

perceptron_accuracy = accuracy_score(y_test, perceptron_best.predict(X_test))
mlp_accuracy = train_and_test_mlp(X_train, y_train, X_test, y_test)

print("Accuracy for Perceptron:", perceptron_accuracy)
print("Accuracy for MLPClassifier:", mlp_accuracy)

def load_data(excel_path):
    df = pd.read_excel(excel_path)
    features = df.drop('Disease', axis=1).to_numpy()
    labels = df['Disease'].to_numpy()
    return features, labels

# Path to the Excel file containing data
excel_path = "C:/Desktop/4TH SEM/Project/ML/extracted_features_1.xlsx"

# Load data
X, y = load_data(excel_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_test_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    confusion = confusion_matrix(y_test, predictions)
    
    return accuracy, precision, recall, f1, confusion

# Define classifiers
classifiers = {
    'MLPClassifier': MLPClassifier(learning_rate_init=0.05, momentum=0.9, max_iter=500),  # Limiting max_iter for MLPClassifier
    'SVM': SVC(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0),  # Suppressing CatBoost verbose output
    'NaiveBayes': GaussianNB()
}

# Define a function to train and test a classifier and return results
def train_test_and_get_results(name, clf):
    return name, train_and_test_classifier(clf, X_train, y_train, X_test, y_test)

# Train and test each classifier using parallel processing
results = Parallel(n_jobs=-1)(delayed(train_test_and_get_results)(name, clf) for name, clf in classifiers.items())

# Organize results into a dictionary
results_dict = {name: {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1, 'Confusion Matrix': conf} for name, (acc, prec, rec, f1, conf) in results}

# Print each classifier's results separately
for classifier_name, results_data in results_dict.items():
    print(f"Classifier: {classifier_name}")
    for metric, value in results_data.items():
        if metric != 'Confusion Matrix':  # Confusion matrix is too large to print
            print(f"{metric}: {value}")
    print("\n")