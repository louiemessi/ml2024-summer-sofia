import numpy as np
from sklearn.metrics import precision_score, recall_score

N = int(input("Enter the number of points (N): "))

ground_truth = np.zeros(N, dtype=int)
predicted_labels = np.zeros(N, dtype=int)

print(f"Please enter {N} points (x, y):")
for i in range(N):
    x = int(input(f"Point {i + 1} - Enter x (ground truth, 0 or 1): "))
    y = int(input(f"Point {i + 1} - Enter y (predicted class, 0 or 1): "))
    ground_truth[i] = x
    predicted_labels[i] = y

precision = precision_score(ground_truth, predicted_labels)
recall = recall_score(ground_truth, predicted_labels)

print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
