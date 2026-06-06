import numpy as np


class EvolutionaryAlgorithm:
    def __init__(
        self, func, bounds, population_size=30, mutation_rate=0.1, mutation_scale=0.5
    ):
        self.func = func
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.dim = len(bounds)

        self.population = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.population_size, self.dim),
        )
        self.fitness = np.array([self.func(ind) for ind in self.population])

    def select_parents(self):
        parents = []
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(
                self.population_size, size=3, replace=False
            )
            best_index = tournament_indices[np.argmin(self.fitness[tournament_indices])]
            parents.append(self.population[best_index])
        return np.array(parents)

    def crossover(self, parents):
        offspring = np.empty_like(parents)
        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < self.population_size else parents[0]
            alpha = np.random.rand(self.dim)
            offspring[i] = alpha * parent1 + (1 - alpha) * parent2
            if i + 1 < self.population_size:
                offspring[i + 1] = alpha * parent2 + (1 - alpha) * parent1
        return offspring

    def mutate(self, offspring):
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                step = np.random.normal(0, self.mutation_scale, self.dim)
                offspring[i] += step
                offspring[i] = np.clip(
                    offspring[i], self.bounds[:, 0], self.bounds[:, 1]
                )
        return offspring

    def step(self):
        parents = self.select_parents()
        offspring = self.crossover(parents)
        offspring = self.mutate(offspring)
        new_fitness = np.array([self.func(ind) for ind in offspring])
        self.population = offspring
        self.fitness = new_fitness
        best = np.min(self.fitness)
        average = np.mean(self.fitness)
        worst = np.max(self.fitness)
        return best, average, worst, np.copy(self.population)


class DifferentialEvolution:
    def __init__(self, func, bounds, population_size=50, F=0.8, CR=0.9):
        self.func = func
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.dim = len(bounds)

        self.population = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.population_size, self.dim),
        )
        self.fitness = np.array([self.func(ind) for ind in self.population])

    def step(self):
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)

        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = a + self.F * (b - c)
            mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])

            f_trial = self.func(trial)
            if f_trial <= self.fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial
            self.population = new_population
            self.fitness = new_fitness

            best = np.min(self.fitness)
            average = np.mean(self.fitness)
            worst = np.max(self.fitness)
        return best, average, worst, np.copy(self.population)
