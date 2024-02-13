import numpy as np

tsp_data = np.loadtxt("gr17.2085.tsp")
print(tsp_data.shape)

def objective_function(solution, matrix):
    """Calculates the total distance of the path described by the solution."""
    total_distance = 0
    number_of_cities = len(solution)
    for i in range(number_of_cities):
        total_distance += matrix[solution[i-1]][solution[i]]
    return total_distance

def neighbors_generator(solution):
    """Generates all possible pairs of cities to swap and yield new solutions."""
    number_of_cities = len(solution)
    for i in range(number_of_cities):
        for j in range(i + 1, number_of_cities):
            neighbor = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap cities
            yield neighbor

def selection_function(current_solution, matrix):
    """Selects the best neighboring solution."""
    best_distance = objective_function(current_solution, matrix)
    best_neighbor = current_solution
    for neighbor in neighbors_generator(current_solution):
        current_distance = objective_function(neighbor, matrix)
        if current_distance < best_distance:
            best_distance = current_distance
            best_neighbor = neighbor
    return best_neighbor, best_distance

def local_search(matrix):
    """Performs the local search algorithm."""
    # Initial solution (a simple sequence of cities)
    current_solution = [i for i in range(len(matrix))]
    current_distance = objective_function(current_solution, matrix)
    while True:
        new_solution, new_distance = selection_function(current_solution, matrix)
        if new_distance >= current_distance:
            # No improvement, return the current solution
            break
        current_solution, current_distance = new_solution, new_distance
    return current_solution, current_distance



cities = [i for i in range(0, tsp_data.shape[0])]
print(cities)
# Perform the local search algorithm on the TSP data
final_solution, final_distance = local_search(tsp_data)
print(final_solution, final_distance)