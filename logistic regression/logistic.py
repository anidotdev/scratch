from data_gen import X, y, N, sigmoid
from classificationMetrics import ClassificationMetrics
import math

class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_prob(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold = 0.5):
        return 1 if self.predict_prob(x) >= threshold else 0

    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_prob(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n):
                p = self.predict_prob(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | w: [{self.weights[0]:.3f}, {self.weights[1]:.3f}] | b: {self.bias:.3f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)

if __name__ == "__main__":
    split = int(0.8 * N)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("\n=== Training Logistic Regression ===")
    model = LogisticRegression(n_features=2, learning_rate=0.1)
    model.fit(X_train, y_train, epochs=1000, print_every=200)

    print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.4f}")
    print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")
    print(f"Weights: [{model.weights[0]:.4f}, {model.weights[1]:.4f}]")
    print(f"Bias: {model.bias:.4f}")

    y_pred_test = [model.predict(x) for x in X_test]

    metrics = ClassificationMetrics(y_test, y_pred_test)
    metrics.print_report()
