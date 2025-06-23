import numpy as np
import os
import random
import copy
import pickle


class NeuralNetwork:
    """Simple neural network for genetic algorithm"""

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, hidden_size // 2) * 0.1
        self.bias2 = np.zeros((1, hidden_size // 2))
        self.weights3 = np.random.randn(hidden_size // 2, output_size) * 0.1
        self.bias3 = np.zeros((1, output_size))

    def forward(self, x):
        """Forward pass through the network"""
        # Convert input to numpy array if it's not already
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=float)

        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # First hidden layer with ReLU activation
        self.layer1 = np.dot(x, self.weights1) + self.bias1
        self.layer1_activation = np.maximum(0, self.layer1)  # ReLU

        # Second hidden layer with ReLU activation
        self.layer2 = np.dot(self.layer1_activation, self.weights2) + self.bias2
        self.layer2_activation = np.maximum(0, self.layer2)  # ReLU

        # Output layer (no activation for raw scores)
        self.output = np.dot(self.layer2_activation, self.weights3) + self.bias3

        return self.output

    def get_action(self, state):
        """Get action from network output"""
        output = self.forward(state)
        # For binary action (flap/don't flap), use argmax
        action = [0, 0]
        action[np.argmax(output)] = 1
        return action

    def clone(self):
        """Create a deep copy of the network"""
        return copy.deepcopy(self)

    def save(self, file_name='model.pkl'):
        """Save model to file"""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def load(self, file_name='model.pkl'):
        """Load model from file"""
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                loaded_model = pickle.load(f)
                self.weights1 = loaded_model.weights1
                self.bias1 = loaded_model.bias1
                self.weights2 = loaded_model.weights2
                self.bias2 = loaded_model.bias2
                self.weights3 = loaded_model.weights3
                self.bias3 = loaded_model.bias3
            return True
        return False


class GeneticAlgorithm:
    """Genetic algorithm for evolving neural networks"""

    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.7, 
                 elite_size=5, input_size=6, hidden_size=64, output_size=2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.generation = 0

        # Initialize population
        self.population = []
        for _ in range(population_size):
            self.population.append(NeuralNetwork(input_size, hidden_size, output_size))

        # Fitness scores for each individual
        self.fitness_scores = [0] * population_size

    def calculate_fitness(self, scores):
        """Update fitness scores based on game performance"""
        self.fitness_scores = scores

    def select_parents(self):
        """Select parents using tournament selection"""
        parents = []

        # First, add elites
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            parents.append(self.population[idx].clone())

        # Then, use tournament selection for the rest
        while len(parents) < self.population_size:
            # Select random individuals for tournament
            tournament_size = 5
            tournament_indices = random.sample(range(self.population_size), tournament_size)

            # Find the best individual in the tournament
            best_idx = tournament_indices[0]
            for idx in tournament_indices:
                if self.fitness_scores[idx] > self.fitness_scores[best_idx]:
                    best_idx = idx

            # Add the winner to parents
            parents.append(self.population[best_idx].clone())

        return parents

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        if random.random() > self.crossover_rate:
            return parent1.clone()

        child = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)

        # Crossover weights and biases
        # For each layer, randomly choose weights from either parent
        if random.random() < 0.5:
            child.weights1 = parent1.weights1.copy()
            child.bias1 = parent1.bias1.copy()
        else:
            child.weights1 = parent2.weights1.copy()
            child.bias1 = parent2.bias1.copy()

        if random.random() < 0.5:
            child.weights2 = parent1.weights2.copy()
            child.bias2 = parent1.bias2.copy()
        else:
            child.weights2 = parent2.weights2.copy()
            child.bias2 = parent2.bias2.copy()

        if random.random() < 0.5:
            child.weights3 = parent1.weights3.copy()
            child.bias3 = parent1.bias3.copy()
        else:
            child.weights3 = parent2.weights3.copy()
            child.bias3 = parent2.bias3.copy()

        return child

    def mutate(self, individual):
        """Apply random mutations to an individual"""
        # Mutate weights1
        mask = np.random.random(individual.weights1.shape) < self.mutation_rate
        individual.weights1[mask] += np.random.normal(0, 0.1, size=np.sum(mask))

        # Mutate bias1
        mask = np.random.random(individual.bias1.shape) < self.mutation_rate
        individual.bias1[mask] += np.random.normal(0, 0.1, size=np.sum(mask))

        # Mutate weights2
        mask = np.random.random(individual.weights2.shape) < self.mutation_rate
        individual.weights2[mask] += np.random.normal(0, 0.1, size=np.sum(mask))

        # Mutate bias2
        mask = np.random.random(individual.bias2.shape) < self.mutation_rate
        individual.bias2[mask] += np.random.normal(0, 0.1, size=np.sum(mask))

        # Mutate weights3
        mask = np.random.random(individual.weights3.shape) < self.mutation_rate
        individual.weights3[mask] += np.random.normal(0, 0.1, size=np.sum(mask))

        # Mutate bias3
        mask = np.random.random(individual.bias3.shape) < self.mutation_rate
        individual.bias3[mask] += np.random.normal(0, 0.1, size=np.sum(mask))

        return individual

    def evolve(self):
        """Evolve the population to the next generation"""
        # Select parents
        parents = self.select_parents()

        # Create new population through crossover and mutation
        new_population = []

        # First, add elites without modification
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx].clone())

        # Then, create the rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            # Create child through crossover
            child = self.crossover(parent1, parent2)

            # Apply mutation
            child = self.mutate(child)

            # Add to new population
            new_population.append(child)

        # Update population
        self.population = new_population
        self.generation += 1

        # Return the best individual from the previous generation
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx].clone()

    def get_population(self):
        """Return the current population"""
        return self.population

    def get_best_individual(self):
        """Return the best individual from the current population"""
        if max(self.fitness_scores) > 0:
            best_idx = np.argmax(self.fitness_scores)
            return self.population[best_idx].clone()
        else:
            # If no individual has a positive score, return the first one
            return self.population[0].clone()

    def save_best(self, file_name='best_model.pkl'):
        """Save the best individual to a file"""
        best = self.get_best_individual()
        best.save(file_name)
