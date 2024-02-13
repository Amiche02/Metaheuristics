import numpy as np

# Load data from tsp file
tsp_data = np.loadtxt('gr17.2085.tsp')
print(tsp_data.shape)


# Objective function to calculate the total distance of a tour
def objective_fun(x, dist_mat):
    distance = 0
    for i in range(len(x) - 1):
        distance += dist_mat[x[i], x[i + 1]]
    distance += dist_mat[x[-1], x[0]]
    return distance


# Function to generate neighboring solutions
def generate_neighbors(s):
    neighbors = []
    for i in range(1, len(s) - 1):
        for j in range(i + 1, len(s)):
            snew = s.copy()
            snew[i], snew[j] = snew[j], snew[i]
            neighbors.append(snew)
    return neighbors


# Function to generate an initial solution
def get_InitialSolution(dist_mat):
    num_cities = len(dist_mat)
    solution = np.arange(num_cities)
    np.random.shuffle(solution)
    return solution.tolist()


def tabu_Search(dist_mat, iterations, max_tabu_size, intensify=False):
    # Initialize the initial solution and tabu list
    s = get_InitialSolution(dist_mat)
    best_solution = s.copy()
    best_cost = objective_fun(best_solution, dist_mat)
    tabu_list = []
    num_iter_without_improvement = 0

    # Repeat the search for a given number of iterations
    for iteration in range(iterations):
        neighbors = generate_neighbors(s)
        neighbors_cost = np.array([objective_fun(n, dist_mat) for n in neighbors])
        non_tabu_neighbors = [neighbors[i] for i in range(len(neighbors)) if tuple(neighbors[i]) not in tabu_list]

        # If all neighbors are tabu, choose the best one (aspiration criterion)
        if not non_tabu_neighbors:
            non_tabu_neighbors = neighbors
            neighbors_cost = np.array([objective_fun(n, dist_mat) for n in non_tabu_neighbors])

        # Find the best admissible move among non-tabu neighbors
        non_tabu_costs = [objective_fun(n, dist_mat) for n in non_tabu_neighbors]
        best_neighbor_index = np.argmin(non_tabu_costs)
        s = non_tabu_neighbors[best_neighbor_index]
        s_cost = non_tabu_costs[best_neighbor_index]

        # If the new solution is better, update the best solution
        if s_cost < best_cost:
            best_solution = s.copy()
            best_cost = s_cost
            num_iter_without_improvement = 0
        else:
            num_iter_without_improvement += 1

        # Update the tabu list
        tabu_list.append(tuple(s))
        if len(tabu_list) > max_tabu_size:
            tabu_list.pop(0)

        # Intensification process: if no improvement, return to the best solution
        if intensify and num_iter_without_improvement >= int(iterations * 0.1):
            s = best_solution.copy()
            num_iter_without_improvement = 0

    return best_solution, best_cost


# Run the algorithm for both versions
print(f"\nInitial Solution : {get_InitialSolution(tsp_data)}")

# Version 1: Basic Tabu Search with only Tabu list
best_sol_v1, best_obj_v1 = tabu_Search(tsp_data, iterations=100, max_tabu_size=15)
print(f"\nVersion 1 - Best Solution: {best_sol_v1}")
print(f"Version 1 - Objective value: {best_obj_v1}")

# Version 2: Tabu Search with Intensification process
best_sol_v2, best_obj_v2 = tabu_Search(tsp_data, iterations=100, max_tabu_size=15, intensify=True)
print(f"\nVersion 2 - Best Solution: {best_sol_v2}")
print(f"Version 2 - Objective value: {best_obj_v2}")