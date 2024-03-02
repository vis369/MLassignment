import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import librosa
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
def load_data(excel_path):
    # Load data from Excel sheet
    df = pd.read_excel(excel_path)

    # Separate features and labels
    features = df.drop('Disease', axis=1).to_numpy()
    labels = df['Disease'].to_numpy()

    return features, labels

def knn_classifiers(X,y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a KNeighborsClassifier as an example
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)

    # Predictions on training and test sets
    y_train_pred = knn_model.predict(X_train)
    y_test_pred = knn_model.predict(X_test)
    
    return X_train, X_test, y_train, y_test, y_train_pred,y_test_pred

def metrix(y_train,y_train_pred,y_test,y_test_pred):
    # Confusion matrix
    confusion_train = confusion_matrix(y_train, y_train_pred)
    confusion_test = confusion_matrix(y_test, y_test_pred)

    # Performance metrics
    precision_train = precision_score(y_train, y_train_pred, average='weighted')
    recall_train = recall_score(y_train, y_train_pred, average='weighted')
    f1_train = f1_score(y_train, y_train_pred, average='weighted')

    precision_test = precision_score(y_test, y_test_pred, average='weighted')
    recall_test = recall_score(y_test, y_test_pred, average='weighted')
    f1_test = f1_score(y_test, y_test_pred, average='weighted')
    return confusion_train,confusion_test,precision_train,recall_train,f1_train,precision_test,recall_test,f1_test

def rand_train_data():
    np.random.seed(42)  # Set seed for reproducibility
    X_train = np.random.uniform(1, 10, size=(20, 2))
    y_train = np.random.randint(2, size=20)

    # Scatter plot of training data
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue', label='Class 0')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='red', label='Class 1')
    plt.title('Training Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    return X_train, y_train

def rand_test_data(X_train, y_train):
    X_test = np.array(np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))).T.reshape(-1, 2)

    # Classify test points using kNN (k=3)
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)

    # Scatter plot of test data with predicted class colors
    plt.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], color='blue', label='Predicted Class 0')
    plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], color='red', label='Predicted Class 1')
    plt.title('Test Data Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    return X_test
    
def variable_k(X_train, y_train, X_test):
    k_values = [1,2,3]
    for k in k_values:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train, y_train)
        y_pred = knn_classifier.predict(X_test)

        # Scatter plot of test data with predicted class colors for each k
        plt.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], label=f'Predicted Class 0 (k={k})')
        plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], label=f'Predicted Class 1 (k={k})')

        plt.title('Test Data Predictions for Different k Values')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
    


    
excel_path = "C:/Users/vishn/Downloads/extracted_features.xlsx"
X, y = load_data(excel_path)
X_train1, X_test1, y_train1, y_test1, y_train_pred1,y_test_pred1 = knn_classifiers(X,y)

#A1
confusion_train,confusion_test,precision_train,recall_train,f1_train,precision_test,recall_test,f1_test = metrix(y_train1,y_train_pred1,y_test1,y_test_pred1)

print("Confusion Matrix (Training Data):\n", confusion_train)
print("\nPrecision (Training Data):", precision_train)
print("Recall (Training Data):", recall_train)
print("F1-Score (Training Data):", f1_train)

print("\nConfusion Matrix (Test Data):\n", confusion_test)
print("\nPrecision (Test Data):", precision_test)
print("Recall (Test Data):", recall_test)
print("F1-Score (Test Data):", f1_test)

# A3
X_train, y_train = rand_train_data()

# A4
X_test = rand_test_data(X_train, y_train)

# A5
variable_k(X_train, y_train, X_test)



#A7

# Define parameter grid
param_grid = {'n_neighbors': range(1, 11)}  # Example range from 1 to 10

# Initialize kNN classifier
knn = KNeighborsClassifier()

# Instantiate GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5)

# Fit GridSearchCV to data
grid_search.fit(X_train1, y_train1)


# Retrieve best estimator
best_estimator = grid_search.best_estimator_

# Retrieve grid scores
grid_scores = grid_search.cv_results_['mean_test_score']

# Retrieve best parameters and best score
best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_

print(f'Best k: {best_k}')
print(f'Best Accuracy: {best_score}')
print(f'Best Estimator: {best_estimator}')
print(f'Grid Scores: {grid_scores}')


#A2
data_sheet1 = pd.read_excel("C:/Users/vishn/Downloads/Lab Session1 Data.xlsx", sheet_name='Purchase data')

# Step 2: Train-test split
X = data_sheet1[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
y = data_sheet1['Payment (Rs)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training
knn_regressor = KNeighborsRegressor(n_neighbors=3)  # Initialize kNN regressor
knn_regressor.fit(X_train, y_train)  # Train the model

# Step 4: Prediction
y_pred = knn_regressor.predict(X_test)  # Predict prices for the testing data

# Step 5: Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')




np.random.seed(42)
data = {'Feature_X': np.random.uniform(1, 10, 20),
        'Feature_Y': np.random.uniform(1, 10, 20),
        'Class': np.random.randint(2, size=20)}

df = pd.DataFrame(data)

# Separate features and labels
X = df[['Feature_X', 'Feature_Y']].values
y = df['Class'].values

# Scatter plot of training data
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1')
plt.title('Training Data')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.legend()
plt.show()

# A4: Generate test set data
X_test = np.array(np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))).T.reshape(-1, 2)

# Classify test points using kNN (k=3)
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X, y)
y_pred = knn_classifier.predict(X_test)

# Scatter plot of test data with predicted class colors
plt.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], color='blue', label='Predicted Class 0')
plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], color='red', label='Predicted Class 1')
plt.title('Test Data Predictions (k=3)')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.legend()
plt.show()

# A5: Repeat for various values of k
k_values = [1, 5, 10]
for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X, y)
    y_pred = knn_classifier.predict(X_test)

    # Scatter plot of test data with predicted class colors for each k
    plt.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], label=f'Predicted Class 0 (k={k})')
    plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], label=f'Predicted Class 1 (k={k})')

    plt.title(f'Test Data Predictions for Different k Values (k={k})')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.legend()
    plt.show()