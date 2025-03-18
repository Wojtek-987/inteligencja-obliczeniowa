import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("../Dane/iris1.csv")

train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=292571)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]


def classify_iris(sl, sw, pl, pw):
    if pl < 2.45:
        return "Setosa"
    elif pl >= 4.95:
        return "Virginica"
    else:
        return "Versicolor"


# Evaluate the manual classifier
dd_predictions = [classify_iris(row[0], row[1], row[2], row[3]) for row in test_inputs]
dd_accuracy = accuracy_score(test_classes, dd_predictions)
dd_cm = confusion_matrix(test_classes, dd_predictions)

print("Manual Classifier (DD):")
print("Accuracy: {:.2f}%".format(dd_accuracy * 100))
print("Confusion Matrix:")
print(dd_cm)
print()

# Evaluate k-NN classifiers for k = 3, 5 and 11.
for k in [3, 5, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_inputs, train_classes)
    knn_predictions = knn.predict(test_inputs)
    knn_accuracy = accuracy_score(test_classes, knn_predictions)
    knn_cm = confusion_matrix(test_classes, knn_predictions)
    print("k-NN Classifier with k = {}:".format(k))
    print("Accuracy: {:.2f}%".format(knn_accuracy * 100))
    print("Confusion Matrix:")
    print(knn_cm)
    print()

# Evaluate the Naive Bayes classifier.
nb = GaussianNB()
nb.fit(train_inputs, train_classes)
nb_predictions = nb.predict(test_inputs)
nb_accuracy = accuracy_score(test_classes, nb_predictions)
nb_cm = confusion_matrix(test_classes, nb_predictions)

print("Naive Bayes Classifier:")
print("Accuracy: {:.2f}%".format(nb_accuracy * 100))
print("Confusion Matrix:")
print(nb_cm)
