import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import librosa
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def load_audio_features(audio_file, max_length=200):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Extract features using Short-Time Fourier Transform (STFT)
    stft = librosa.stft(y, n_fft=512, hop_length=256)
    magnitudes = np.abs(stft)

    # Pad or truncate features to a fixed length
    if magnitudes.shape[1] < max_length:
        pad_width = max_length - magnitudes.shape[1]
        magnitudes = np.pad(magnitudes, pad_width=((0, 0), (0, pad_width)))
    elif magnitudes.shape[1] > max_length:
        magnitudes = magnitudes[:, :max_length]

    # Flatten the feature matrix to a 1D array
    flattened_features = magnitudes.flatten()

    return flattened_features

# Load data from Excel sheet
excel_path = "C:/Users/vishn/OneDrive - Amrita vishwa vidyapeetham/Desktop/ML Course/archive/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv"
df = pd.read_csv(excel_path)

# Initialize empty lists to store features and labels
features = []
labels = []

# Path to the folder containing audio and annotation files
data_folder = "C:/Users/vishn/OneDrive - Amrita vishwa vidyapeetham/Desktop/ML Course/archive/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"

# Create a dictionary to map patient IDs to diseases
patient_disease_mapping = dict(zip(df['Patient_ID'], df['Disease']))
# Set a maximum length for the features
max_feature_length = 200

# Iterate through all files in the folder
for file_name in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file_name)

    # Check if the file is a .wav or .txt file
    if file_name.endswith('.wav'):
        # Extract patient ID from the file name
        patient_id = int(file_name[:3])

        # Check if patient ID is in the mapping
        if patient_id in patient_disease_mapping:
            disease = patient_disease_mapping[patient_id]

            # Load audio features using STFT
            audio_features = load_audio_features(file_path, max_length=max_feature_length)

            # Append the features and labels
            features.append(audio_features)
            labels.append(disease)

# Convert lists to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data (e.g., scale the features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Plot histogram for training data
plt.figure(figsize=(10, 6))
plt.hist(y_train, bins=np.arange(len(np.unique(y_train)) + 1) - 0.5, alpha=0.7, rwidth=0.8)
plt.xticks(np.arange(len(np.unique(y_train))))
plt.title('Histogram - Training Data')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for testing data
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=np.arange(len(np.unique(y_test)) + 1) - 0.5, alpha=0.7, rwidth=0.8)
plt.xticks(np.arange(len(np.unique(y_test))))
plt.title('Histogram - Testing Data')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# A1
# Calculate mean for each class (class centroid)
class_centroids = []
for label in np.unique(y_train):
    class_features = X_train[y_train == label]
    centroid = np.mean(class_features, axis=0)
    class_centroids.append(centroid)

# Calculate spread (standard deviation) for each class
class_spreads = []
for label in np.unique(y_train):
    class_features = X_train[y_train == label]
    spread = np.std(class_features, axis=0)
    class_spreads.append(spread)

# Calculate distance between mean vectors between classes
interclass_distances = []
for i in range(len(class_centroids)):
    for j in range(i + 1, len(class_centroids)):
        distance = np.linalg.norm(class_centroids[i] - class_centroids[j])
        interclass_distances.append(distance)

# Print results
print("Class Centroids:")
for i, centroid in enumerate(class_centroids):
    print("Class {}: {}".format(i, centroid))
print("\nClass Spreads:")
for i, spread in enumerate(class_spreads):
    print("Class {}: {}".format(i, spread))
print("\nInterclass Distances:")
for i, distance in enumerate(interclass_distances):
    print("Distance between Class {} and Class {}: {}".format(i, i+1, distance))



# A2
# Select a feature (for example, the first feature)
selected_feature_index = 0
selected_feature = X_train[:, selected_feature_index]

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(selected_feature, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Selected Feature')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.show()

# Calculate mean and variance
feature_mean = np.mean(selected_feature)
feature_variance = np.var(selected_feature)

print("Mean of the selected feature:", feature_mean)
print("Variance of the selected feature:", feature_variance)


#A3
# Select two feature vectors
feature_vector_1 = X_train[0]  # Selecting the first feature vector
feature_vector_2 = X_train[1]  # Selecting the second feature vector

# Calculate Minkowski distances for r from 1 to 10
r_values = range(1, 11)
minkowski_distances = []

for r in r_values:
    distance = minkowski(feature_vector_1, feature_vector_2, r)
    minkowski_distances.append(distance)

# Plot the distances
plt.figure(figsize=(10, 6))
plt.plot(r_values, minkowski_distances, marker='o', linestyle='-')
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.xticks(r_values)
plt.grid(True)
plt.show()


#A4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

#A5
# Initialize the kNN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the classifier using the training data
neigh.fit(X_train, y_train)

#A6
# Test the accuracy of the kNN classifier using the test data
accuracy = neigh.score(X_test, y_test)
print("Accuracy of the kNN classifier on the test set:", accuracy)

#A7
# Predict class labels for the test data
predicted_labels = neigh.predict(X_test)

# Display the predicted labels
print("Predicted labels for the test set:")
print(predicted_labels)

#A8
# Vary k from 1 to 11 and make an accuracy plot for kNN classifiers
k_values = range(1, 12)
accuracies_knn = []

for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    accuracy = neigh.score(X_test, y_test)
    accuracies_knn.append(accuracy)

# Plot accuracy vs k for kNN classifiers
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies_knn, marker='o', linestyle='-')
plt.title('Accuracy vs. k for kNN Classifiers')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()  


# Train the kNN classifier with k=3
neigh_k3 = KNeighborsClassifier(n_neighbors=3)
neigh_k3.fit(X_train, y_train)

# Function to calculate and print evaluation metrics
def evaluate_performance(y_true, y_pred, dataset_name):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix for", dataset_name, "data:")
    print(cm)

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("Precision for", dataset_name, "data:", precision)
    print("Recall for", dataset_name, "data:", recall)
    print("F1-Score for", dataset_name, "data:", f1)
    print()

# Evaluate performance on training data
train_predicted_labels = neigh_k3.predict(X_train)
evaluate_performance(y_train, train_predicted_labels, "Training")

# Evaluate performance on test data
test_predicted_labels = neigh_k3.predict(X_test)
evaluate_performance(y_test, test_predicted_labels, "Test")
