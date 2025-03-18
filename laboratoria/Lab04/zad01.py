import math


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def forwardPass(wiek: float, waga: float, wzrost: float) -> float:

    # Hidden layer weights and biases
    # Hidden neuron 1
    w_h1_wiek = -0.46122
    w_h1_waga = 0.97314
    w_h1_wzrost = -0.39203
    b_h1 = 0.80109

    # Hidden neuron 2
    w_h2_wiek = 0.78548
    w_h2_waga = 2.10584
    w_h2_wzrost = -0.57847
    b_h2 = 0.43529

    # Output layer weights and bias
    w_out_h1 = -0.81546
    w_out_h2 = 1.03775
    b_out = -0.2368

    # ------------------------

    hidden1 = (w_h1_wiek * wiek
               + w_h1_waga * waga
               + w_h1_wzrost * wzrost
               + b_h1)
    hidden2 = (w_h2_wiek * wiek
               + w_h2_waga * waga
               + w_h2_wzrost * wzrost
               + b_h2)

    hidden1_activated = sigmoid(hidden1)
    hidden2_activated = sigmoid(hidden2)

    output = (w_out_h1 * hidden1_activated
              + w_out_h2 * hidden2_activated
              + b_out)

    return output


def main():
    # Example 1
    wiek_1, waga_1, wzrost_1 = 23, 75, 176
    result_1 = forwardPass(wiek_1, waga_1, wzrost_1)
    print(f"forwardPass({wiek_1}, {waga_1}, {wzrost_1}) = {result_1:.6f}")

    # Example 2
    wiek_2, waga_2, wzrost_2 = 50, 68, 180
    result_2 = forwardPass(wiek_2, waga_2, wzrost_2)
    print(f"forwardPass({wiek_2}, {waga_2}, {wzrost_2}) = {result_2:.6f}")


if __name__ == "__main__":
    main()
