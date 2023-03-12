import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# Read CSV, make the dataframe, drop the last two because they aren't full length
data = pd.read_csv('blink_1_averaged.csv', header=0)
data = pd.DataFrame(data)

# Data features
x_data = data.iloc[:, :-1]
x_data = np.array(data.iloc[:, :-1].values)

# Data labels
y_data = data.iloc[:, -1]
y_data = np.array(data.iloc[:, -1].values)

# Split into test and training data
x_train, x_test, y_train, y_test = train_test_split(  # type: ignore
    x_data, y_data, test_size=0.2, shuffle=False, random_state=42
)

# Normalize the data
x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)

# ----- Generating plots -----
plt.subplot(1,2,1)
for i in range(10):
    plt.plot(x_train[i*2,:])
plt.subplot(1,2,2)
for i in range(10):
    plt.plot(x_train[i*2+1,:])
plt.show()

# ----- MODEL -----
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

cv_scores = []

# Loop over each fold
for train_index, test_index in kf.split(x_data):
    # Split the data into training and testing sets
    X_train, X_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    # Train the model on the training set
    model = SVC()
    model.fit(X_train, y_train)

    # Evaluate the model on the testing set and store the score
    score = model.score(X_test, y_test)
    cv_scores.append(score)

# Calculate the mean and standard deviation of the cross-validation scores
mean_score = np.mean(cv_scores)
std_score = np.std(cv_scores)

# Print the mean and standard deviation of the cross-validation scores
print("Cross-validation scores: ", cv_scores)
print("Mean score: ", mean_score)