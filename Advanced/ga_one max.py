# Genetic Algorithms: One-Max Problem
# Single-Point Crossover
import random


def random_genome(string_length):
    # Generate a random string with a given length
    return [random.choice([0, 1]) for _ in range(string_length)]


def fitness(genome):
    # Find fitness: The number of 1s in the string
    fitness_list = []
    for i in range(len(genome)):
        # Create a sublist that only stores the value 1 in the list
        genome_one = [genome[i][j] for j in range(len(genome[i]))
                      if genome[i][j] == 1]
        # Calculate its length and append it to list
        fitness = len(genome_one)
        fitness_list.append(fitness)
    return fitness_list


def choose_parent(genome, top_parents):
    fitness_list = fitness(genome)
    # Merge The Genome String and its fitness to a List of Tuples
    merged_list = [(genome[i], fitness_list[i]) for i in range(len(genome))]
    sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)
    # Choose top two tuples
    return sorted_list[:top_parents]


def crossover(genome, top_parents):
    # Get the Parent String
    parent1, parent2 = choose_parent(genome, top_parents)
    parent1, parent2 = parent1[0], parent2[0]
    # Get the crossover point, use len(genome) because of slicing rule
    cross_point = random.randint(0, len(genome))
    # Get Offspring
    offspring1 = parent1[:cross_point] + parent2[cross_point:]
    offspring2 = parent2[:cross_point] + parent1[cross_point:]
    return offspring1, offspring2


def mutation(genome, top_parents, mutation_rate):  # Bit flips from 0 to 1
    # if mutation_rate is less than 0.1 -> (1 - bit), else retain the bit
    offspring1, offspring2 = crossover(genome, top_parents)
    offspring1_flip = [bit if random.random() > mutation_rate else 1 - bit for bit in offspring1]
    offspring2_flip = [bit if random.random() > mutation_rate else 1 - bit for bit in offspring2]
    return offspring1_flip, offspring2_flip


def new_generation(genome, top_parents, mutation_rate):
    fitness_list = fitness(genome)
    offspring1, offspring2 = list(mutation(genome, top_parents, mutation_rate))
    # Replace the new genomes
    # Sort current population by fitness to get the top features
    # Because we attempted to change their genome
    merged_list = [(genome[i], fitness_list[i]) for i in range(len(genome))]
    sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)
    # Replace the top_parents with the offspring
    sorted_genome = [gen[0] for gen in sorted_list]  # Get the sorted_genome
    top_genome = [offspring1] + [offspring2]
    sorted_genome = top_genome + [geno for geno in sorted_genome if geno not in sorted_genome[:top_parents]]
    return sorted_genome


def is_solution_found(genome, string_length):
    # Check if any genome is all 1s
    for g in genome:
        if sum(g) == string_length:
            return True, g
    return False, []


def main():
    print("One Max Problem")
    string_length = 5
    population_size = 4
    mutation_rate = 1 / string_length
    generations = 100
    top_parents = 2
    # Initialize population
    genome = [random_genome(string_length) for _ in range(population_size)]
    for gen in range(generations):
        print(f"\nGeneration {gen + 1}:")
        for i, individual in enumerate(genome):
            print(f"- {i+1}: {individual} | Fitness: {sum(individual)}")
        found, solution = is_solution_found(genome, string_length)
        if found:
            print(f"\nSolution found in generation {gen + 1}: {solution}")
            break
        genome = new_generation(genome, top_parents, mutation_rate)
    else:
        print("\nSolution not found within the generation limit.")


if __name__ == "__main__":
    main()
