from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # (d) Original network: single hidden layer of 2 neurons
    mlp_2neurons = MLPClassifier(
        hidden_layer_sizes=(2,),
        activation='relu',
        max_iter=3000,
        random_state=42
    )
    mlp_2neurons.fit(X_train_scaled, y_train)
    acc_2neurons = mlp_2neurons.score(X_test_scaled, y_test)

    # (f) Single hidden layer of 3 neurons
    mlp_3neurons = MLPClassifier(
        hidden_layer_sizes=(3,),
        activation='relu',
        max_iter=3000,
        random_state=42
    )
    mlp_3neurons.fit(X_train_scaled, y_train)
    acc_3neurons = mlp_3neurons.score(X_test_scaled, y_test)

    # (g) Two hidden layers, each with 3 neurons
    mlp_2layers_3neurons = MLPClassifier(
        hidden_layer_sizes=(3, 3),
        activation='relu',
        max_iter=3000,
        random_state=42
    )
    mlp_2layers_3neurons.fit(X_train_scaled, y_train)
    acc_2layers_3neurons = mlp_2layers_3neurons.score(X_test_scaled, y_test)

    # Print accuracies for comparison
    print(f"Accuracy with single hidden layer (2 neurons): {acc_2neurons:.2f}")
    print(f"Accuracy with single hidden layer (3 neurons): {acc_3neurons:.2f}")
    print(f"Accuracy with two hidden layers (3 neurons each): {acc_2layers_3neurons:.2f}")


if __name__ == "__main__":
    main()
