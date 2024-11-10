import numpy as np

class KNNRegression:
    def __init__(self, N, k, points):
        # N: number of data points
        # k: number of neighbors
        # points: list of (x, y) points as numpy array
        self.N = N
        self.k = k
        self.points = np.array(points)

    def calculate_distance(self, x1, x2):
        # Calculate Euclidean distance between x1 and x2
        return np.abs(x1 - x2)

    def predict(self, x_input):
        # Check if k is less than or equal to N
        if self.k > self.N:
            raise ValueError("k cannot be greater than N")

        # Calculate the distances between the input and all points
        distances = []
        for point in self.points:
            distance = self.calculate_distance(point[0], x_input)
            distances.append((distance, point[1]))

        # Sort the distances and take the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = distances[:self.k]

        # Calculate the prediction by averaging the Y values of the k nearest neighbors
        prediction = np.mean([neighbor[1] for neighbor in nearest_neighbors])
        return prediction

def main():
    # Get user inputs
    try:
        N = int(input("Enter the number of points (N): "))
        if N <= 0:
            print("N must be a positive integer.")
            return
        
        k = int(input("Enter the number of neighbors (k): "))
        if k <= 0:
            print("k must be a positive integer.")
            return
        
        # Reading points (x, y)
        points = []
        for i in range(N):
            x = float(input(f"Enter x value for point {i + 1}: "))
            y = float(input(f"Enter y value for point {i + 1}: "))
            points.append((x, y))

        # Create a KNNRegression object
        knn = KNNRegression(N, k, points)

        # Get input for prediction
        x_input = float(input("Enter the x value for prediction: "))
        
        # Get the result from k-NN regression
        y_pred = knn.predict(x_input)
        
        print(f"The predicted Y value for X = {x_input} is: {y_pred}")
        
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
