import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2,1), random.gauss(2,1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5,1), random.gauss(5,1)])
    y.append(1)

combined = list(zip(X,y))
random.shuffle(combined)
X, y = zip(*combined)

X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print("Class 0 center: (2, 2), Class 1 center: (5, 5)")
print("First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")

if __name__ == "__main__":
    print(f"Generated {N} samples")
    print(f"X shape: {len(X)} x {len(X[0])}")
    print(f"y length: {len(y)}")

    print("\nFirst 5 samples:")
    for i in range(5):
        print(f"Features: {X[i]}, Label: {y[i]}")
