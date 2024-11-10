import numpy as np
import random

num_faces = 6  
grid_size = 6  
MAX_STAGNATION = 9000

def generate_initial_population(pop_size, initial_puzzle):
    population = []
    for _ in range(pop_size):
        individual = np.copy(initial_puzzle)
        for face in range(num_faces):
            for i in range(grid_size):
                empty_indices = np.where(individual[face, i, :] == 0)[0]
                np.random.shuffle(empty_indices)
                possible_values = np.arange(1, grid_size + 1)
                np.random.shuffle(possible_values)
                individual[face, i, empty_indices] = possible_values[:len(empty_indices)]
        population.append(individual)
    return population

def calculate_fitness(individual):
    fitness = 0
    
    # Restrição 1: Unicidade em linhas e colunas para cada face
    for face in range(num_faces):
        for i in range(grid_size):
            # Penaliza duplicatas em linhas e colunas
            fitness += (grid_size - len(set(individual[face, i, :])))  # Linha
            fitness += (grid_size - len(set(individual[face, :, i])))  # Coluna
    
    # Restrição 2: Unicidade em subgrades 3x3 dentro de cada face
    for face in range(num_faces):
        for row in range(0, grid_size, 3):
            for col in range(0, grid_size, 3):
                subgrid = individual[face, row:row+3, col:col+3].flatten()
                fitness += (grid_size - len(set(subgrid)))
    
    # Restrição 3: Unicidade ao longo do eixo Z
    for i in range(grid_size):       
        for j in range(grid_size):    
            z_line = [individual[face, i, j] for face in range(num_faces)]
            fitness += (grid_size - len(set(z_line)))  # Penaliza duplicatas ao longo do eixo Z
    
    return fitness


# Crossover principal
def crossover(parent1, parent2):
    child = np.copy(parent1)
    for face in range(num_faces):
        if random.random() > 0.5:
            child[face] = parent2[face]
    return child

# Crossover com heurística (Seleciona apenas melhor fitness)
def selective_crossover(parent1, parent2):
    child = np.copy(parent1)
    for face in range(num_faces):
        for i in range(grid_size):
            if np.random.rand() > 0.5:
                child[face, i, :] = parent2[face, i, :]
            else:
                child[face, :, i] = parent2[face, :, i]
    return child

def mutate(individual, mutation_rate=0.01):
    for face in range(num_faces):
        if random.random() < mutation_rate:
            row = random.randint(0, grid_size - 1)
            idx1, idx2 = random.sample(range(grid_size), 2)
            individual[face, row, idx1], individual[face, row, idx2] = individual[face, row, idx2], individual[face, row, idx1]
    return individual

def genetic_algorithm(pop_size, max_generations, initial_puzzle):
    population = generate_initial_population(pop_size, initial_puzzle)
    best_fitness_over_time = []

    stagnation = 0
    last_best_fitness = float('inf')

    for generation in range(max_generations):
        population.sort(key=calculate_fitness)
        best_fitness = calculate_fitness(population[0])
        if best_fitness <= last_best_fitness:
            stagnation += 1
        else:
            stagnation = 0

        best_fitness_over_time.append(best_fitness)
        
        print(f"Geração {generation}: Melhor fitness = {best_fitness}")
        
        if stagnation >= MAX_STAGNATION:
            print("Não foi possível encontrar melhor solução. Estagnação por mais de 2000 gerações")
            return population[0]

        if best_fitness == 0:
            print(f"Solução encontrada na geração {generation}")
            return population[0]
        
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(population[:pop_size // 2], 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
    
    print("Nenhuma solução encontrada")
    return None

initial_puzzle = np.zeros((num_faces, grid_size, grid_size), dtype=int)

solution = genetic_algorithm(pop_size=50, max_generations=1000000, initial_puzzle=initial_puzzle)
if solution is not None:
    print("Solução final:")
    print(solution)

