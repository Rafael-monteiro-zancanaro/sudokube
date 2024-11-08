import numpy as np
import random

# Configurações do SudoKube
num_faces = 6  # Número de faces do cubo
grid_size = 9  # Tamanho do grid 9x9 para cada face

# Função para gerar uma população inicial respeitando pistas fixas
def generate_initial_population(pop_size, initial_puzzle):
    population = []
    for _ in range(pop_size):
        individual = np.copy(initial_puzzle)
        for face in range(num_faces):
            for i in range(grid_size):
                # Preenche apenas células vazias
                empty_indices = np.where(individual[face, i, :] == 0)[0]
                np.random.shuffle(empty_indices)
                possible_values = np.arange(1, grid_size + 1)
                np.random.shuffle(possible_values)
                individual[face, i, empty_indices] = possible_values[:len(empty_indices)]
        population.append(individual)
    return population

# Função de fitness aprimorada
def calculate_fitness(individual):
    fitness = 0
    for face in range(num_faces):
        for i in range(grid_size):
            # Penaliza duplicatas em linhas e colunas
            fitness += (grid_size - len(set(individual[face, i, :])))
            fitness += (grid_size - len(set(individual[face, :, i])))
        
        # Verifica duplicatas em cada subgrade 3x3
        for row in range(0, grid_size, 3):
            for col in range(0, grid_size, 3):
                subgrid = individual[face, row:row+3, col:col+3].flatten()
                fitness += (grid_size - len(set(subgrid)))
    
    # Adiciona penalidade por inconsistências nas bordas compartilhadas
    fitness += check_shared_edges(individual)
    
    return fitness

# Função para verificar as bordas compartilhadas entre as faces
def check_shared_edges(individual):
    penalty = 0
    # Adicione a lógica para comparar as bordas de faces adjacentes
    # Por exemplo, comparar o topo de uma face com a base da outra
    # Aqui deve ser implementado o código específico para o layout do SudoKube
    return penalty

# Operador de crossover
def crossover(parent1, parent2):
    child = np.copy(parent1)
    for face in range(num_faces):
        if random.random() > 0.5:
            child[face] = parent2[face]  # Substitui a face com a do outro pai
    return child

# Operador de mutação
def mutate(individual, mutation_rate=0.01):
    for face in range(num_faces):
        if random.random() < mutation_rate:
            # Troca aleatória em uma linha para mutação
            row = random.randint(0, grid_size - 1)
            idx1, idx2 = random.sample(range(grid_size), 2)
            individual[face, row, idx1], individual[face, row, idx2] = individual[face, row, idx2], individual[face, row, idx1]
    return individual

# Algoritmo genético principal
def genetic_algorithm(pop_size, max_generations, initial_puzzle):
    population = generate_initial_population(pop_size, initial_puzzle)
    best_fitness_over_time = []

    for generation in range(max_generations):
        population.sort(key=calculate_fitness)
        best_fitness = calculate_fitness(population[0])
        best_fitness_over_time.append(best_fitness)
        
        print(f"Geração {generation}: Melhor fitness = {best_fitness}")
        
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

# Exemplo de input: um SudoKube com pistas fixas
initial_puzzle = np.zeros((num_faces, grid_size, grid_size), dtype=int)
# Preencha 'initial_puzzle' com as pistas do problema antes de rodar o algoritmo

# Executa o algoritmo genético
solution = genetic_algorithm(pop_size=50, max_generations=1000000, initial_puzzle=initial_puzzle)
if solution is not None:
    print("Solução final:")
    print(solution)

