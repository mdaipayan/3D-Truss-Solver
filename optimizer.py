import numpy as np
import copy
import random

class TrussOptimizerGA:
    def __init__(self, base_truss, member_groups, pop_size=50, generations=100, 
                 crossover_rate=0.8, mutation_rate=0.1, 
                 min_area=1e-5, max_area=0.01, 
                 yield_stress=172.3e6, density=2768.0):
        """
        Initializes the Genetic Algorithm for Truss Optimization.
        Default material properties are set for the 25-bar/72-bar Aluminum benchmark.
        """
        self.base_truss = base_truss
        self.member_groups = member_groups  # Dictionary: {0: [member_ids], 1: [member_ids]}
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.min_area = min_area
        self.max_area = max_area
        self.yield_stress = yield_stress
        self.density = density  # kg/m^3 (Objective is to minimize weight)
        
        self.num_vars = len(member_groups)
        
        # Initialize Population (Random cross-sectional areas within bounds)
        self.population = np.random.uniform(self.min_area, self.max_area, (self.pop_size, self.num_vars))
        self.fitness = np.zeros(self.pop_size)
        
        # Tracking the best results
        self.best_chromosome = None
        self.best_fitness = float('inf')
        self.convergence_history = []

    def evaluate_individual(self, chromosome):
        """
        Maps a chromosome (design variables) to the truss, solves it, 
        and calculates Weight + Stress Penalty.
        """
        # Create a temporary copy of the truss so we don't overwrite the base model
        temp_truss = copy.deepcopy(self.base_truss)
        
        # 1. Apply the chromosome (areas) to the corresponding member groups
        for group_idx, member_ids in self.member_groups.items():
            area = chromosome[group_idx]
            for m in temp_truss.members:
                if m.id in member_ids:
                    m.A = area
                    # Update local stiffness matrix with new area
                    m.k_global_matrix = (m.E * m.A / m.L) * np.outer(m.T_vector, m.T_vector)
                    
        # 2. Solve the structural system
        try:
            temp_truss.solve()
        except ValueError:
            # If the matrix is singular/unstable, apply an infinite penalty
            return float('inf')
            
        # 3. Calculate Objective (Weight) and Constraints (Stress Violations)
        total_weight = 0.0
        max_stress_violation = 0.0
        
        for m in temp_truss.members:
            # Weight = Area * Length * Density
            total_weight += m.A * m.L * self.density
            
            # Stress = Internal Force / Area
            stress = abs(m.internal_force / m.A)
            
            # Calculate penalty if stress exceeds the yield limit
            if stress > self.yield_stress:
                # Normalized violation percentage
                violation = (stress - self.yield_stress) / self.yield_stress
                max_stress_violation = max(max_stress_violation, violation)
                
        # 4. Fitness Formulation = Weight + Penalty
        # A massive penalty multiplier ensures the GA heavily favors safe designs
        penalty_multiplier = 1e6  
        fitness = total_weight + (max_stress_violation * penalty_multiplier)
        
        return fitness

    def select_parents(self):
        """Tournament Selection: Pick the best of 3 random individuals."""
        tournament_size = 3
        idx = np.random.choice(self.pop_size, tournament_size, replace=False)
        best_idx = idx[np.argmin(self.fitness[idx])]
        return self.population[best_idx]

    def crossover(self, parent1, parent2):
        """Simulated Binary Crossover (SBX) or Simple Blend."""
        if random.random() < self.crossover_rate:
            alpha = random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            return child1, child2
        return np.copy(parent1), np.copy(parent2)

    def mutate(self, child):
        """Gaussian Mutation: Slightly tweak areas to explore new designs."""
        for i in range(self.num_vars):
            if random.random() < self.mutation_rate:
                # Mutate by up to +/- 10% of the max area
                mutation_step = np.random.normal(0, 0.1 * (self.max_area - self.min_area))
                child[i] += mutation_step
                
                # Enforce physical boundaries
                child[i] = np.clip(child[i], self.min_area, self.max_area)
        return child

    def run(self):
        """Executes the Genetic Algorithm evolution loop."""
        for gen in range(self.generations):
            
            # 1. Evaluate Current Population
            for i in range(self.pop_size):
                self.fitness[i] = self.evaluate_individual(self.population[i])
                
                # Update global best
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_chromosome = np.copy(self.population[i])
                    
            self.convergence_history.append(self.best_fitness)
            
            # 2. Create Next Generation
            new_population = []
            
            # Elitism: Automatically keep the absolute best design
            new_population.append(self.best_chromosome)
            
            while len(new_population) < self.pop_size:
                p1 = self.select_parents()
                p2 = self.select_parents()
                
                c1, c2 = self.crossover(p1, p2)
                
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)
                    
            self.population = np.array(new_population)
            
        return self.best_chromosome, self.best_fitness, self.convergence_history
