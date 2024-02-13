import numpy as np

def objective_function(s):
    x = 0
    for i, elt in enumerate(s):
        x += elt*(2**(len(s)-(i+1)))

    return x**3 - 60*x**2 + 900*x + 100

def random_neighborhood_generator(s_initial):
    idx = np.random.randint(0, len(s_initial))
    temp = s_initial.copy()
    temp[idx] = 1 - temp[idx]
    return temp

def simulated_annealing(initial_solution, initial_temperature, min_temperature, alpha):
    current_solution = initial_solution
    current_temperature = initial_temperature
    best_solution = current_solution
    best_objective = objective_function(best_solution)
    i = 0

    while current_temperature > min_temperature:
        neighbor = random_neighborhood_generator(current_solution)
        delta_e = objective_function(neighbor) - objective_function(current_solution)
        probability = np.exp(-abs(delta_e) / current_temperature)
        rand = np.random.rand()
        equilibrium = 0

        print(f"Solution {i} : {best_solution, best_objective}")

        while equilibrium <= 5:
            if delta_e > 0 or rand < probability: # random.choices([True, False], [probability, 1-probability])
                current_solution = neighbor

                current_objective = objective_function(current_solution)
                if current_objective > best_objective:
                    best_solution = current_solution
                    best_objective = current_objective
            equilibrium += 1
            print(f"random : {rand}  ---> Probability : {probability}")
        i += 1
        current_temperature *= alpha
    return best_solution, best_objective

s_init = 10011

s_init = int(s_init)
s_init = [int(c) for c in str(s_init)]


print(f"\nBest solution : {simulated_annealing(s_init, 500, 200, 0.9)}")
#print(random_neighborhood_generator(s_init))