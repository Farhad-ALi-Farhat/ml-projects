import numpy as np

def compute_cost(X, y, w, b):
    m = len(y)
    y_pred = X * w + b
    cost = (1/(2*m)) * np.sum((y_pred - y) ** 2)
    return cost

def gradient_descent(X, y, w, b, lr, epochs):
    m = len(y)

    for i in range(epochs):
        y_pred = X * w + b

        # Gradients
        dw = (1/m) * np.sum((y_pred - y) * X)
        db = (1/m) * np.sum(y_pred - y)

        # Update
        w = w - lr * dw
        b = b - lr * db

        if i % 100 == 0:
            print(f"Epoch {i}, Cost: {compute_cost(X, y, w, b):.4f}")

    return w, b

def main():
    # Simple dataset: y = 2x
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    w = 0
    b = 0
    lr = 0.01
    epochs = 1000

    w, b = gradient_descent(X, y, w, b, lr, epochs)

    print("\nLearned parameters:")
    print("w:", w)
    print("b:", b)

    print("Prediction for x=6:", w * 6 + b)

if __name__ == "__main__":
    main()
