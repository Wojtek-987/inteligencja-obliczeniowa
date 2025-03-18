import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    data = pd.read_csv("../Dane/diabetes1.csv")

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split into training (70%) and test (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(6, 3),
        activation='relu',
        max_iter=500,
        random_state=42
    )

    mlp.fit(X_train_scaled, y_train)

    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy on the test set: {acc:.2f}")
    print("Confusion matrix:")
    print(cm)

    # new network
    mlp_new = MLPClassifier(
        hidden_layer_sizes=(8, 4),
        activation='tanh',
        max_iter=500,
        random_state=42
    )
    mlp_new.fit(X_train_scaled, y_train)
    y_pred_new = mlp_new.predict(X_test_scaled)
    acc_new = accuracy_score(y_test, y_pred_new)
    cm_new = confusion_matrix(y_test, y_pred_new)

    print("\n=== New Network (8,4) with tanh ===")
    print(f"Accuracy: {acc_new:.2f}")
    print("Confusion Matrix:")
    print(cm_new)

    """
        f) Które błędy są gorsze w diagnozie cukrzycy: FP czy FN?

        FP: przewidywanie cukrzycy u zdrowej osoby (mniej groźne, bo tylko dodatkowe testy).
        FN: niewykrycie cukrzycy u chorej osoby (poważniejsze, bo opóźnia diagnozę i leczenie).
        Zwykle FN są bardziej niebezpieczne. Jeśli w macierzy błędu jest dużo FN, to niepokojące.


        Sieć ma więcej FN niż FP:
        [127  23]
        [ 36  45]
        oraz
        [128  22]
        [ 31  50]
    """


if __name__ == "__main__":
    main()
