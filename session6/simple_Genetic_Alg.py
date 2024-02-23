import numpy as np

# Parameters
population_size = 100
chromosome_length = 20
mutation_rate = 0.01
crossover_rate = 0.7
num_generations = 100

# Initialize population
population = np.random.randint(2, size=(population_size, chromosome_length))
print(f"Population : {population} \n\nShape : {population.shape}")

def fitness(individual):
    return np.sum(individual)

def select(population):
    fitnesses = np.array([fitness(individual) for individual in population])
    return population[np.random.choice(range(population_size), size=population_size, replace=True, p=fitnesses/fitnesses.sum())]

def crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, chromosome_length-1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2

    return parent1, parent2

def mutate(individual):
    for i in range(chromosome_length):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm():
    global population
    for generation in range(num_generations):
        new_population = []
        selected_population = select(population)
        for i in range(0, population_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = np.array(new_population)
        best_fitness = np.max([fitness(individual) for individual in population])
        print(f"Generation {generation+1}: Best Fitness = {best_fitness}")
    best_individual = population[np.argmax([fitness(individual) for individual in population])]
    return best_individual

best_solution = genetic_algorithm()
print("Best Solution:", best_solution)

