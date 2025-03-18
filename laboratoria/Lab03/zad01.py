import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../Dane/iris1.csv")
print("Full dataset:")
print(df)

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=292571)

print("\nTest set:")
print(test_set)
print("Number of records in test set:", test_set.shape[0])


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


good_predictions = 0
length = test_set.shape[0]

for i in range(length):
    if classify_iris(test_set[i, 0], test_set[i, 1], test_set[i, 2], test_set[i, 3]) == test_set[i, 4]:
        good_predictions += 1

accuracy = (good_predictions / length) * 100

print("\nNumber of correct predictions:", good_predictions, "out of", length)
print("Accuracy: {:.2f}%".format(accuracy))
