import numpy as np
import random
import pandas as pd
from dataclasses import dataclass, field
from typing import List
import argparse
import os

### PARAMETERS AND SETTINGS

subpops = 10                     # Number of subgrids
individuals_per_subpop = 100     # Number of individuals in each subgrid
generations = 2                  # Number of generations
language_length = 5              # Length of the language vector
num_examples = 5                 # Number of examples for communication
mu = 0.05                        # Mutation rate for bias
mut_strength = 0.3               # Standard deviation for bias mutation
migration_rate = 0.05            # Probability of migration between subgrids
s = 1                            # Selective strength on communicative success
trade_off = 0.001                # Cost weight for regularity bias             
decreasing_use = False           # System for decreasing use of subsequent language elements either true or false
individual_differences = False   # System to allow differences for all individuals that are different in the population
different_prob = 0               # Probability to belong to the "different" group
adjustments = 0                  # Number of adjustments to bias post learning (if 0, no adjustments are made, only learning)

sim_id = 0      # Simulation ID for output file naming

### DATA STRUCTURES

@dataclass
class Individual:
    language: np.ndarray = field(default_factory=lambda: np.zeros(language_length, dtype=float))  # Array of floats [0, 1]
    bias: float = 0.0               # Non-negative float ranging from 0 to 1
    different: int = 0              # 0 or 1
    population: int = 0             # Index of the subpopulation the individual belongs to
    production: np.ndarray = field(default_factory=lambda: np.zeros(language_length, dtype=int))
    fitness: float = 0.0

    def __eq__(self, value:  "Individual") -> bool:
        return (isinstance(value, Individual) and
                np.array_equal(self.language, value.language) and
                self.bias == value.bias and
                self.different == value.different and
                self.population == value.population)
        

### INITIALIZATION

def initialize_population() -> List[Individual]:
    population = []
    for subpop in range(subpops):
        subpopulation = []
        for _ in range(individuals_per_subpop):
            language = np.random.rand(language_length)  # Random values between 0 and 1
            bias = random.uniform(0, 1)                 # Random bias between 0 and 1
            different = 1 if random.random() < different_prob else 0
            individual = Individual(language=language, bias=bias, different=different, population=subpop)
            subpopulation.append(individual)
        population.append(subpopulation)
    return population

if decreasing_use:
    usage_weights = np.linspace(1, 1/language_length, language_length)
    print("decreasing use is enabled, usage weights: ", usage_weights)

### FUNCTIONS

def production(individual: Individual) -> np.ndarray:
    # np.random.binomial processes the whole language array at once
    return np.random.binomial(1, individual.language)

def communication_success(receiver: Individual, population: List[Individual]) -> float:
    receiver_examples = production(receiver)
    
    # Select unique producers using random.sample (faster than checking 'if producer not in producers')
    # Note: If population size can be less than num_examples, you'll need to handle that edge case.
    producers = random.sample(population, min(num_examples, len(population)))
    
    # Get all producer examples into a 2D array
    producer_examples = np.array([production(p) for p in producers])
    
    if decreasing_use:
        # Array comparison: Which elements match? (Creates a boolean matrix)
        matches = (producer_examples == receiver_examples)
        
        # Create a mask based on usage_weights (simulating random.random() < usage_weights)
        use_mask = np.random.rand(*matches.shape) < usage_weights
        
        # Only count matches where the element was actually used
        number_of_successful_communications = np.sum(matches & use_mask)
        number_of_total_communications = np.sum(use_mask)
    else:
        # Simple vectorized array comparison
        matches = (producer_examples == receiver_examples)
        number_of_successful_communications = np.sum(matches)
        number_of_total_communications = matches.size
    
    return number_of_successful_communications / number_of_total_communications if number_of_total_communications > 0 else 1 # 100% success if no communication attempts were made (rare edge case) --> CHECK BEFORE UPLOAD

def evaluate_fitness(population: List[Individual]) -> float:
    # Calculate fitness of each individual based on communicative success and bias
    for individual in population:
        comm_success = communication_success(individual, population)  # Communicative success with the whole population as producers
        bias_cost = trade_off * individual.bias  # Regularity bias cost
        individual.fitness = s * comm_success + bias_cost
    return

def learning(individual: Individual, parents: List[Individual]) -> None:
    #Randomly sample parents and generate their productions directly into a 2D array
    examples = np.array([production(random.choice(parents)) for _ in range(num_examples)])

    a = individual.bias
    b = a
    epsilon = 1e-9
    
    #Sum down the columns (axis=0) to get 'k' for all language elements simultaneously
    k = np.sum(examples, axis=0)
    
    #Calculate alpha and beta parameters as entire arrays
    alpha_params = np.maximum(k + a, epsilon)
    beta_params = np.maximum(num_examples - k + b, epsilon)

    #Generate the entire new language vector at once
    individual.language = np.random.beta(alpha_params, beta_params)

def adjusting_lang(individual: Individual, parents: List[Individual]) -> None:
    examples = np.array([production(random.choice(parents)) for _ in range(num_examples)])
    epsilon = 1e-9
    
    # Calculate base 'a' and 'b' arrays based on current language
    # Assuming the "100" logic you mentioned in your comments
    a_base = 100 * individual.language 
    b_base = 100 - a_base
    
    k = np.sum(examples, axis=0)
    
    alpha_params = np.maximum(k + a_base, epsilon)
    beta_params = np.maximum(num_examples - k + b_base, epsilon)
    
    individual.language = np.random.beta(alpha_params, beta_params)
    
def reproduce(population: List[Individual]) -> List[Individual]:
    # Reproduction of subpopulation based on fitness, with possibility for mutation
    # Watch out for population size so it stays constant
    fitness = [max(0.0, a.fitness) for a in population]  # Extract fitness values, ensuring they are non-negative
    total = sum(fitness)  # Calculate total fitness across all agents in subpopµ
    children = []  # Initialize list to store offspring agents
    if total <= 0:  # If all fitness values are zero or negative
        if len(population) == 0:
            return children
        else:
            probs = [1.0 / len(population)] * len(population)  # Select uniformly at random
    else:  # If there is positive total fitness
        probs = [f / total for f in fitness]  # Normalize fitness values to get selection probabilities
    cum = []  # Initialize list for cumulative probabilities
    acc = 0.0  # Initialize accumulator for cumulative sum
    for p in probs:  # Iterate through each probability
        acc += p  # Add current probability to accumulator
        cum.append(acc)  # Append cumulative probability to list
    # Sort subpopulation by fitness (descending order)
    for _ in range(individuals_per_subpop):  # Create as many children as there are parents
        r = random.random()  # Generate random number for roulette wheel selection
        for i, c in enumerate(cum):  # Iterate through cumulative probabilities
            if r < c:  # If random number falls below this cumulative probability
                parent = population[i]  # Select the corresponding parent
                new_bias = parent.bias  # Inherit bias from parent
                if random.random() < mu:  # Apply mutation to bias with probability mu
                    new_bias += np.random.normal(0, mut_strength)  # Add small random value to bias
                    #new_bias = max(0.0, min(1.0, new_bias))  # Ensure bias stays within [0, 1] # CHECK BEFORE UPLOAD
                new_different = random.choice([0, 1])
                new_agent = Individual(bias=new_bias, different=new_different, population=parent.population)  # Create new agent with inherited mutated bias and empty language (to be learned later)
                children.append(new_agent)  # Add the new agent to the children list
                break  # Break the loop after selecting one parent
    return children

def migrate(population: List[Individual]) -> List[Individual]:
    # Initialize a clean structure for the migrated population
    next_population = [[] for _ in range(subpops)]
    
    for i, subpop in enumerate(population):
        for individual in subpop:
            if random.random() < migration_rate:
                # Calculate new subpopulation index with wrap-around
                new_subpop = (i + random.choice([-1, 1])) % subpops
                individual.population = new_subpop
                next_population[new_subpop].append(individual)
            else:
                # Individual stays in their current subpopulation
                next_population[i].append(individual)
                
    return next_population

### MAIN SIMULATION

pop = initialize_population()
data = []  # To store data for analysis
print("Initialization complete, starting simulation...")

for gen in range(generations):
    # For each generation, perform reproduction, migration, and learning
    print(f"Generation {gen}/{generations}", end="\r")
    children = []
    for i, subpop in enumerate(pop):
        evaluate_fitness(subpop)
        data.append({
            'generation': gen,
            'subpopulation': i,
            'average_fitness': np.mean([ind.fitness for ind in subpop]),
            'average_bias': np.mean([ind.bias for ind in subpop]),
            'average_language_1': np.mean([ind.language[0] for ind in subpop], axis=0),
            'average_language_2': np.mean([ind.language[1] for ind in subpop], axis=0),
            'average_language_3': np.mean([ind.language[2] for ind in subpop], axis=0),
            'average_language_4': np.mean([ind.language[3] for ind in subpop], axis=0),
            'average_language_5': np.mean([ind.language[4] for ind in subpop], axis=0)
            })
        if subpop == []:
            children.append([])
            continue  # Skip empty subpopulations
            
        subpopchildren = reproduce(subpop)
        for individual in subpopchildren:
            learning(individual, subpop)  # Learning from the current subpopulation as parents
            if adjustments > 0:
                for _ in range(adjustments):
                    adjusting_lang(individual, subpop)  # Adjust language post learning
        children.append(subpopchildren)  # Add children to the subpopulation
    data.append({
        'generation': gen,
        'subpopulation': "total",
        'average_fitness': np.mean([ind.fitness for subpop in pop for ind in subpop]),
        'average_bias': np.mean([ind.bias for subpop in pop for ind in subpop]),
        'average_language_1': np.mean([ind.language[0] for subpop in pop for ind in subpop], axis=0),
        'average_language_2': np.mean([ind.language[1] for subpop in pop for ind in subpop], axis=0),
        'average_language_3': np.mean([ind.language[2] for subpop in pop for ind in subpop], axis=0),
        'average_language_4': np.mean([ind.language[3] for subpop in pop for ind in subpop], axis=0),
        'average_language_5': np.mean([ind.language[4] for subpop in pop for ind in subpop], axis=0)
        })
    pop = migrate(children)  # Perform migration between subpopulations, this is the new starting population for the next generation



### DATA ANALYSIS

df = pd.DataFrame(data)
csv_path = "simulation_base_{sim_id}.csv"
df.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path} - Replicate {sim_id} complete!")

