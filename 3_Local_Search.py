#Local Search Algorithm (LS
import numpy as np

def objective_function(s):
    x = 0
    for i, elt in enumerate(s):
        x += elt*(2**(len(s)-(i+1)))

    return x**3 - 60*x**2 + 900*x, x

def neighborhood_generator(s_initial):
    s_initial = [int(c) for c in str(s_initial)]
    neighbors = []
    for i in range(len(s_initial)):
        temp = s_initial.copy()
        temp[i] = 1 - temp[i]
        neighbors.append(temp)
    return neighbors

def selection_function(functions):
    functions = np.array(functions)
    return np.argmax(functions)

s = 10001
neighbors_ = neighborhood_generator(s)
neighbors_.insert(0, [int(c) for c in str(s)])
print(f"Neighbors : {neighbors_}")

functions_ = []
decimals = []
for elt in neighbors_:
    function, x = objective_function(elt)
    functions_.append(function)
    decimals.append(x)

best_solution_idx = selection_function(functions_)

print(f"Decimal values : {decimals}\nFunctions : {functions_}")
print(f"Best Solution : {neighbors_[best_solution_idx]}\nBest function value : {functions_[best_solution_idx]}")






