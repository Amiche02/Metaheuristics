import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tsp_problem import GeneticAlgorithm
import numpy as np

tsp_data = np.loadtxt('..\gr17.2085.tsp')
print(tsp_data.shape)

def objective_function(s):
    cost = 0
    for i in range(s.shape[0] - 1):
        cost = cost + tsp_data[s[i]][s[i + 1]]

    cost = cost + tsp_data[s[-1]][s[0]]
    return cost

# Since we need to track the evolution of the tours over generations, we need to modify the GA to store the best tour of each generation
class GeneticAlgorithmWithHistory(GeneticAlgorithm):
    def run_with_history(self):
        history = []
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
            current_best_fitness = min([self.fitness(individual) for individual in self.population])
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = self.population[np.argmin([self.fitness(individual) for individual in self.population])]
                history.append(best_individual)
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {-best_fitness}")
        return best_individual, -best_fitness, history

# Initialize the Genetic Algorithm with history tracking
ga = GeneticAlgorithmWithHistory(tsp_data)

# Run the Genetic Algorithm with history tracking
best_tour, best_distance, history = ga.run_with_history()
print("Best Tour:", best_tour)
print("Best Distance:", best_distance)

# Function to plot the TSP tour
def plot_tsp(tour, ax, distance_matrix):
    ax.clear()
    ax.set_title(f"Total Distance: {-ga.fitness(tour):.2f}")
    start_point = tour[0]
    ax.plot(distance_matrix[start_point, 0], distance_matrix[start_point, 1], 'o', markerfacecolor='red')
    for i in range(1, len(tour)):
        start_point = tour[i-1]
        end_point = tour[i]
        ax.plot([distance_matrix[start_point, 0], distance_matrix[end_point, 0]],
                [distance_matrix[start_point, 1], distance_matrix[end_point, 1]], 'k-')
    ax.plot(distance_matrix[end_point, 0], distance_matrix[end_point, 1], 'o', markerfacecolor='blue')

# Set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()

# Since we don't have the actual coordinates, we'll use the indices as coordinates for visualization purposes
coordinates = np.array([(i, i) for i in range(len(tsp_data))])

# Animation function which updates the plot with the current tour
def animate(i):
    tour = history[i]
    plot_tsp(tour, ax, coordinates)

# Call the animator, blit=True means only re-draw the parts that have changed
anim = animation.FuncAnimation(fig, animate, frames=len(history), interval=500, blit=False, repeat=False)

plt.close()  # Prevents duplicate display

# Display the animation
HTML(anim.to_html5_video())
