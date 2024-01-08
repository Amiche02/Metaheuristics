# Function to calculate factorial
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


# Function to calculate the number of tours possible
def calculate_tours(n, optimized=False):
    if optimized:
        return int(0.5 * factorial(n - 1))
    else:
        return factorial(n)

# Calculate and print the number of tours and estimated time for each city count
for n in range(5, 21):
    num_tours = calculate_tours(n)
    time_est =num_tours/1000
    print(f"Number of cities: {n}, Number of tours: {num_tours}, Estimated time: {time_est}")

    # Also calculate with optimization
    num_tours_optimized = calculate_tours(n, optimized=True)
    time_est_optimized = num_tours_optimized/1000
    print(
        f"Number of cities (Optimized): {n}, Number of tours: {num_tours_optimized}, Estimated time: {time_est_optimized}")
