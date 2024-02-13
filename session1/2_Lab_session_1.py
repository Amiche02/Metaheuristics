import numpy as np
import itertools

def combination_paths(cities_distance):
    return list(itertools.permutations(cities_distance))

def calculate_distances(combination):
    distances = []
    for elt in combination:
        distance = 0
        for i in range(len(elt)-1):
            distance += np.abs(elt[i] - elt[i+1])
        #distance += np.abs(elt[0] - elt[-1])
        distances.append(distance)
    return distances


coordinates = []
#read_file = open("test.txt", "r")
with open("p01.15.291.tsp", "r") as r:
    for line in r.readlines():
        parts = line.strip().split("        ")
        coordinates.append(parts)
#print(coordinates)

cities = np.array(coordinates[0], dtype=np.float32)
#print(cities)

combinations = combination_paths(cities[:5])
distances = calculate_distances(combinations)
min_index = np.argmin(distances)
print(f"Paths : {cities[: 5]}")
print(f"\nMinimum index : {min_index}")
print(f"\nMinimum distance : {distances[min_index]}\nOptimal Combination : {combinations[min_index]}")
