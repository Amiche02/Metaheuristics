import numpy as np

tsp_data = np.loadtxt("gr17.2085.tsp")
print(tsp_data.shape)

def objective_fun(x):
    # Convert the binary vector into a decimal value
    dec = 0
    for i, bit in enumerate(x):
        dec += bit * (2 ** (len(x) - (i + 1)))

    # Calculate the objective function value
    s = (dec ** 3) - (60 * (dec ** 2)) + (900 * dec)
    return s


def generate_neighbors(s_initial):
    neighbors = []
    for i in range(len(s_initial)):
        temp = s_initial.copy()
        temp[i] = 1 - temp[i]
        neighbors.append(temp)
    return neighbors


def best_neighbor(s0):
    # The selection strategy for the best neighbor improvement
    f0 = objective_fun(s0)
    neighbors = generate_neighbors(s0)
    objs = np.apply_along_axis(objective_fun, axis=1, arr=neighbors)
    f = objs[np.argmax(objs)]
    s = neighbors[np.argmax(objs)]
    if f <= f0:
        return s0, f0, False
    else:
        return s, f, True

def first_best_neighbor(s0):
    # The selecting strategy for the first best neighbor improvement
    f0 = objective_fun(s0)
    for i in range(len(s0)):
        neighbor = generate_neighbors(s0, i)
        f = objective_fun(neighbor)
        if f > f0:
            return neighbor, f, True

    return s0, objective_fun(s0), False

cities = [i for i in range(0, tsp_data.shape[0])]
print(cities)