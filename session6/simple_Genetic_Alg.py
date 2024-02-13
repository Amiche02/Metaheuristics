import numpy as np

# Define parameters for the GA
POPULATION_SIZE = 100
GENOME_LENGTH = 20
GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01

# Initialize random population
np.random.seed(42)  # for reproducible results
population = np.random.randint(2, size=(POPULATION_SIZE, GENOME_LENGTH))

# Fitness function: counts the number of 1s in the genome
def fitness(genome):
    return np.sum(genome)

# Selection function: tournament selection
def select(population, tournament_size=5):
    best = None
    for _ in range(tournament_size):
        individual = population[np.random.randint(len(population))]
        if best is None or fitness(individual) > fitness(best):
            best = individual
    return best

# Crossover function: single point crossover
def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        crossover_point = np.random.randint(GENOME_LENGTH)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    else:
        return parent1, parent2

# Mutation function: flip bits
def mutate(genome):
    for i in range(len(genome)):
        if np.random.rand() < MUTATION_RATE:
            genome[i] = 1 - genome[i]
    return genome

# Main GA loop
for generation in range(GENERATIONS):
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        # Select parents
        parent1 = select(population)
        parent2 = select(population)
        # Crossover parents to create children
        child1, child2 = crossover(parent1, parent2)
        # Mutate children
        child1 = mutate(child1)
        child2 = mutate(child2)
        # Add children to the new population
        new_population.append(child1)
        new_population.append(child2)
    population = np.array(new_population)

    # Evaluate the fitness of the new population
    fitness_values = np.array([fitness(individual) for individual in population])
    best_individual = population[np.argmax(fitness_values)]
    best_fitness = fitness(best_individual)

    # Print the best fitness in the population
    print(f"Generation {generation}: Best Fitness = {best_fitness}")

# Output the best individual and its fitness
best_fitness = np.max(fitness_values)
best_individual = population[np.argmax(fitness_values)]
(best_individual, best_fitness)
