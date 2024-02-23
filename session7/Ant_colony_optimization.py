import numpy as np

tsp_data = np.loadtxt('..\gr17.2085.tsp')
print(tsp_data.shape)


def objective_function(s, tsp_data):
    cost = 0
    for i in range(len(s) - 1):
        cost += tsp_data[s[i], s[i + 1]]
    cost += tsp_data[s[-1], s[0]]
    return cost


def ant_colony_optimization(tsp_data, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, q=100):

    num_cities = tsp_data.shape[0]

    # Initialize pheromones: τ
    pheromones = np.ones((num_cities, num_cities))
    # Heuristic information: η, inverse of the tsp distance
    heuristic_info = 1 / (tsp_data + np.diag([np.inf] * num_cities))

    # Keep track of the best solution found
    best_cost = np.inf
    best_solution = None

    for iteration in range(num_iterations):
        # Array to hold all the solutions generated in this iteration
        all_solutions = []
        all_costs = []

        for ant in range(num_ants):
            # Each ant builds a solution
            solution = []
            visited = set()
            # Start with a random city
            current_city = np.random.randint(num_cities)
            solution.append(current_city)
            visited.add(current_city)

            # Construct the rest of the solution
            for i in range(num_cities - 1):
                probabilities = (pheromones[current_city] ** alpha) * (heuristic_info[current_city] ** beta)
                probabilities[list(visited)] = 0
                probabilities /= probabilities.sum()
                next_city = np.random.choice(num_cities, p=probabilities)

                solution.append(next_city)
                visited.add(next_city)
                current_city = next_city

            # Evaluate the solution
            cost = objective_function(solution, tsp_data)
            all_solutions.append(solution)
            all_costs.append(cost)

            # Update the best solution if necessary
            if cost < best_cost:
                best_cost = cost
                best_solution = solution

        # Update pheromones
        pheromones *= (1 - evaporation_rate)  # Evaporation
        for solution, cost in zip(all_solutions, all_costs):
            for i in range(num_cities - 1):
                pheromones[solution[i], solution[i + 1]] += q / cost
            # Update from last city to first to complete the tour
            pheromones[solution[-1], solution[0]] += q / cost

    return best_solution, best_cost


# We will test the implementation with a smaller number of iterations and ants for quick execution.
test_solution, test_cost = ant_colony_optimization(tsp_data, num_ants=5, num_iterations=20)
print(test_solution, test_cost)
