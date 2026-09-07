# Knapsack Problem
# Improved with Genetic Algorithm
import random
import matplotlib.pyplot as plt

items = [[7, 5], [2, 4], [1, 7], [9, 2], [5, 1], [3, 3]]
capacity = 15
generations = 100
population_size = 10
mutation_rate = 0.05


# Initial population: Randomly generated.
def initial_population(items):
    return [random.choice([0, 1]) for _ in range(len(items))]


# Fitness: The total profit of selected if they do not exceed the capacity
def fitness(genomes):
    fitness_list = []  # Fitness in this case is the profit of the solution
    # that does not exceed the capacity
    genomes_weight = [[items[j][0] if genomes[i][j] == 1 else 0
                       for j in range(len(genomes[i]))] for i in range(len(genomes))]
    # This genome store the profit
    genomes_profit = [[items[j][1] if genomes[i][j] == 1 else 0
                       for j in range(len(genomes[i]))] for i in range(len(genomes))]
    # Get the total weight and total profit
    for gen in range(len(genomes)):
        total_weight, total_profit = sum(genomes_weight[gen]), sum(genomes_profit[gen])
        # Only keep those that satisfies the weight capacity
        if capacity >= total_weight > 0:
            fitness_list.append((genomes[gen], total_profit))
    return fitness_list


def reproduction(genomes):  # the top
    # 10% of the current population is kept as-is for next generation.
    fitness_list = fitness(genomes)
    # Sort the Fitness list based on the fitness
    sorted_list = sorted(fitness_list, key=lambda x: x[1], reverse=True)
    # Choose top 10% fitness
    # If the list is empty, return 1 as the maximum
    top_parents = max(2, len(sorted_list) // 10)
    # Return the top parent
    return sorted_list[:top_parents]


def crossover(genomes):
    #  the top 50% of current population is selected to generate 90% of the next population:
    #  Probability of crossover from parent1 = 50%
    #  Probability of crossover from parent2 = 50%
    parents = reproduction(genomes)
    parents = [parent[0] for parent in parents]
    offspring = []
    while len(offspring) < int(population_size * 0.9):
        parent1, parent2 = random.sample(parents, 2)  # Select two random parents
        child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))]
        # If there is mutation, flip the child
        mutated_child = mutation(child, mutation_rate)
        offspring.append(mutated_child)
    return offspring


def mutation(genomes, mutation_rate):
    # Mutation: Probability of the child being mutated = 5%
    return [bit if random.random() > mutation_rate else 1 - bit for bit in genomes]


def main():
    print("Knapsack Problem: Genetic Algorithm")
    global generations
    genomes = [initial_population(items) for _ in range(population_size)]
    best_solution = None
    best_profit = 0
    avg_fitness_history = []

    for generation in range(generations):
        fitness_list = fitness(genomes)
        if fitness_list:
            top_solution = max(fitness_list, key=lambda x: x[1])  # Get the best solution of the generation
            avg_fitness = sum(fit[1] for fit in fitness_list) / len(fitness_list)
            avg_fitness_history.append(avg_fitness)
            # Update best found solution
            if top_solution[1] > best_profit:
                best_solution, best_profit = top_solution
            print(f"Generation {generation + 1}: Best Profit = {top_solution[1]}, Solution = {top_solution[0]}")
            print(f"Average Fitness {avg_fitness}")

        # Generate new population
        genomes = crossover(genomes)

    print(f"\nBest Overall Solution: {best_solution}\nProfit: {best_profit}")

    # Plot the average fitness over generations
    plt.plot(range(1, generations + 1), avg_fitness_history, marker="o", linestyle="-")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness Evolution")
    plt.show()


main()
