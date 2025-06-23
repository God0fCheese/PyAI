import random
import numpy as np
from flappy_game_ai import FlappyGame
from model import NeuralNetwork, GeneticAlgorithm
from helper import plot
import pygame

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITE_SIZE = 5
INPUT_SIZE = 6
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2

class GeneticAgent:
    def __init__(self, population_size=POPULATION_SIZE):
        self.population_size = population_size
        self.generation = 0
        self.best_score = 0
        self.best_fitness = 0

        # Initialize genetic algorithm
        self.ga = GeneticAlgorithm(
            population_size=population_size,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
            elite_size=ELITE_SIZE,
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE
        )

        # Try to load best model
        self.best_model = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        if not self.best_model.load():
            print("No existing model found, starting fresh")

    def get_actions(self, game):
        """Get actions for all birds in the game"""
        actions = []

        # Get state and action for each bird
        for i in range(len(game.birds)):
            state = game.get_state(i)

            # If bird is alive, get action from its neural network
            if game.birds[i].alive:
                # Get the neural network for this bird
                network = self.ga.population[i]
                action = network.get_action(state)
            else:
                # Default action for dead birds
                action = [1, 0]  # Don't flap

            actions.append(action)

        return actions

    def evolve_population(self, fitness_scores):
        """Evolve the population based on fitness scores"""
        # Update fitness scores in genetic algorithm
        self.ga.calculate_fitness(fitness_scores)

        # Save the best model if it's better than previous best
        current_best_fitness = max(fitness_scores)
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.ga.save_best('best_model.pkl')
            print(f"New best fitness: {self.best_fitness:.2f} - Saving model...")

        # Evolve to next generation
        self.ga.evolve()
        self.generation += 1


def train():
    # Initialize game and agent
    game = FlappyGame(population_size=POPULATION_SIZE)
    agent = GeneticAgent(population_size=POPULATION_SIZE)

    # Statistics for plotting
    plot_scores = []
    plot_mean_scores = []
    plot_max_scores = []
    total_score = 0

    # Main training loop
    running = True
    while running:
        # Get actions for all birds
        actions = agent.get_actions(game)

        # Play one step of the game
        fitness_scores, game_over, scores = game.play_step(actions)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # If game is over (all birds are dead), evolve and reset
        if game_over:
            # Evolve population
            agent.evolve_population(fitness_scores)

            # Calculate statistics
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)

            # Update best score
            if max_score > agent.best_score:
                agent.best_score = max_score
                print(f"New best score: {agent.best_score}")

            # Print generation stats
            print(f"Generation: {agent.generation}, Max Score: {max_score}, Avg Score: {avg_score:.2f}, Best Score: {agent.best_score}")

            # Update plots
            plot_scores.append(avg_score)
            plot_max_scores.append(max_score)
            total_score += avg_score
            mean_score = total_score / agent.generation if agent.generation > 0 else 0
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Reset game for next generation
            game.reset_birds(new_generation=True)


if __name__ == "__main__":
    train()
