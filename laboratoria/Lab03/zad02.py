import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

df = pd.read_csv("../Dane/iris1.csv")
print("Complete dataset:")
print(df)

train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=292571)

print("\nTraining Set:")
print(train_set)
print("\nTest Set:")
print(test_set)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

clf = DecisionTreeClassifier(random_state=292571)

# Train the classifier
clf.fit(train_inputs, train_classes)

# Display the decision tree in textual form
tree_rules = export_text(clf, feature_names=list(df.columns[:4]))
print("\nDecision Tree Rules:")
print(tree_rules)

# Display the decision tree graphically
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=list(df.columns[:4]), class_names=clf.classes_, filled=True)
plt.title("Decision Tree for Iris Classification")
plt.show()

# Evaluate the classifier
predictions = clf.predict(test_inputs)
accuracy = accuracy_score(test_classes, predictions)
print("\nAccuracy on Test Set: {:.2f}%".format(accuracy * 100))

# Display the confusion matrix
cm = confusion_matrix(test_classes, predictions)
print("\nConfusion Matrix:")
print(cm)
