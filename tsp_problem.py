import numpy as np

tsp_data = np.loadtxt('..\gr17.2085.tsp')
print(tsp_data.shape)

def objective_function(s):
    cost = 0
    for i in range(s.shape[0] - 1):
        cost = cost + tsp_data[s[i]][s[i + 1]]

    cost = cost + tsp_data[s[-1]][s[0]]
    return cost

# Genetic Algorithm components with the new objective function
class GeneticAlgorithm:
    def __init__(self, distance_matrix, population_size=100, mutation_rate=0.01, crossover_rate=0.7, num_generations=100):
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.num_cities = distance_matrix.shape[0]
        self.population = [np.random.permutation(self.num_cities) for _ in range(population_size)]

    def fitness(self, individual):
        return -objective_function(individual)

    def select(self):
        fitnesses = np.array([self.fitness(individual) for individual in self.population])
        fitnesses -= fitnesses.min()
        if fitnesses.sum() == 0:
            return self.population
        probabilities = fitnesses / fitnesses.sum()
        return list(np.array(self.population)[np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=probabilities)])

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            crossover_point1, crossover_point2 = sorted(np.random.choice(range(self.num_cities), 2, replace=False))
            child = -np.ones(self.num_cities, dtype=int)
            child[crossover_point1:crossover_point2+1] = parent1[crossover_point1:crossover_point2+1]
            for i in range(self.num_cities):
                if child[i] == -1:
                    for j in range(self.num_cities):
                        if parent2[j] not in child:
                            child[i] = parent2[j]
                            break
            return child

        return parent1

    def mutate(self, individual):
        for i in range(self.num_cities):
            if np.random.rand() < self.mutation_rate:
                j = np.random.randint(self.num_cities)
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    def run(self):
        best_fitness = float('inf')
        best_individual = None
        for generation in range(self.num_generations):
            new_population = []
            selected_population = self.select()
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected_population[i], selected_population[i+1]
                child1 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    child2 = self.crossover(parent2, parent1)
                    child2 = self.mutate(child2)
                    new_population.append(child2)
            self.population = new_population
            current_best_fitness = max([self.fitness(individual) for individual in self.population])
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = self.population[np.argmax([self.fitness(individual) for individual in self.population])]
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {-best_fitness}")
        return best_individual, -best_fitness

# Initialize the Genetic Algorithm with the loaded TSP data
ga = GeneticAlgorithm(tsp_data)

# Run the Genetic Algorithm
best_tour, best_distance = ga.run()
print("Best Tour:", best_tour)
print("Best Distance:", best_distance)