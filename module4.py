# Simple program to find the index of a number in a list

def main():
    # Ask the user for input N (positive integer)
    N = int(input("Enter a positive integer N: "))
    
    # Initialize an empty list to store the numbers
    numbers = []

    # Ask the user to provide N numbers one by one
    for i in range(N):
        number = int(input(f"Enter number {i + 1}: "))
        numbers.append(number)

    # Ask the user for input X (integer)
    X = int(input("Enter the integer X to find: "))

    # Check if X is in the list and get the index
    if X in numbers:
        index = numbers.index(X) + 1  # Convert to 1-based index
        print(index)
    else:
        print(-1)

# Run the main function
if __name__ == "__main__":
    main()
