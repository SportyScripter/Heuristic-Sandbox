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

        # Generujemy początkową populację w całkowicie losowych miejscach w granicach mapy
        self.population = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.population_size, self.dim),
        )
        # Oceniamy każdego osobnika (liczymy jego wynik z funkcji)
        self.fitness = np.array([self.func(ind) for ind in self.population])

    def select_parents(self):
        """Wybiera rodziców za pomocą selekcji turniejowej."""
        parents = []
        for _ in range(self.population_size):
            # Losujemy 3 przypadkowych osobników do turnieju
            tournament_indices = np.random.choice(
                self.population_size, size=3, replace=False
            )
            # Wygrywa ten, który ma najmniejszy (najlepszy) wynik
            best_index = tournament_indices[np.argmin(self.fitness[tournament_indices])]
            parents.append(self.population[best_index])
        return np.array(parents)

    def crossover(self, parents):
        """Krzyżowanie (wymiana informacji między rodzicami)."""
        offspring = np.empty_like(parents)
        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            # Zabezpieczenie na wypadek nieparzystej liczby populacji
            parent2 = parents[i + 1] if i + 1 < self.population_size else parents[0]

            # Losujemy wagę (alpha) dla każdego wymiaru osobno, żeby dzieci nie leżały
            # dokładnie na linii prostej między rodzicami
            alpha = np.random.rand(self.dim)
            offspring[i] = alpha * parent1 + (1 - alpha) * parent2

            if i + 1 < self.population_size:
                offspring[i + 1] = alpha * parent2 + (1 - alpha) * parent1

        return offspring

    def mutate(self, offspring):
        """Mutacja (losowe zmiany w dzieciach)."""
        for i in range(self.population_size):
            # Rzucamy kostką - czy w ogóle mutować to dziecko?
            if np.random.rand() < self.mutation_rate:
                # Rozkład normalny (Gaussa) - zazwyczaj robi małe kroki, rzadziej duże
                step = np.random.normal(0, self.mutation_scale, self.dim)
                offspring[i] += step

                # Upewniamy się, że po mutacji osobnik nie uciekł poza granice mapy
                offspring[i] = np.clip(
                    offspring[i], self.bounds[:, 0], self.bounds[:, 1]
                )
        return offspring

    def step(self):
        """Pojedyncza iteracja algorytmu ewolucyjnego."""
        # --- ELITARYZM ---
        # Zapisujemy najlepszego osobnika, żeby go nie zniszczyć w tej iteracji
        best_idx = np.argmin(self.fitness)
        elite_ind = self.population[best_idx].copy()

        parents = self.select_parents()
        offspring = self.crossover(parents)
        offspring = self.mutate(offspring)

        # Wklejamy naszego zwycięzcę na pierwsze miejsce do nowych dzieci
        offspring[0] = elite_ind

        # Oceniamy nową populację
        new_fitness = np.array([self.func(ind) for ind in offspring])

        # Zastępujemy starą populację nową
        self.population = offspring
        self.fitness = new_fitness

        # Wyciągamy statystyki do wykresów
        best = np.min(self.fitness)
        average = np.mean(self.fitness)
        worst = np.max(self.fitness)
        return best, average, worst, np.copy(self.population)


class DifferentialEvolution:
    def __init__(self, func, bounds, population_size=50, F=0.8, CR=0.9):
        self.func = func
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.F = F  # Mnożnik siły mutacji (skala różnicy)
        self.CR = CR  # Prawdopodobieństwo wzięcia zmutowanego genu (crossover rate)
        self.dim = len(bounds)

        # Inicjalizacja tak samo jak w klasycznym AE
        self.population = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.population_size, self.dim),
        )
        self.fitness = np.array([self.func(ind) for ind in self.population])

    def step(self):
        """Pojedyncza iteracja algorytmu Ewolucji Różnicowej (DE)."""
        # Tworzymy brudnopis na nowe wyniki, żeby nie nadpisywać ich w trakcie pętli
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)

        for i in range(self.population_size):
            # Bierzemy wszystkie indeksy oprócz naszego aktualnego osobnika (i)
            idxs = [idx for idx in range(self.population_size) if idx != i]

            # Losujemy 3 zupełnie innych osobników (a, b, c)
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]

            # Główny wzór DE: do wektora 'a' dodajemy przeskalowaną różnicę między 'b' i 'c'
            mutant = a + self.F * (b - c)
            mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

            # Losujemy, które wymiary (osie X, Y, Z) podmieniamy na te od mutanta
            cross_points = np.random.rand(self.dim) < self.CR

            # Zabezpieczenie: musi podmienić chociaż jeden wymiar, inaczej nic by się nie zmieniło
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True

            # Sklejamy próbnego osobnika (tam gdzie True bierzemy od mutanta, reszta od oryginału)
            trial = np.where(cross_points, mutant, self.population[i])

            # Oceniamy go. Jeśli jest lepszy (lub taki sam), zastępuje starego w nowej populacji
            f_trial = self.func(trial)
            if f_trial <= self.fitness[i]:
                new_population[i] = trial
                new_fitness[i] = f_trial

        # Po przetworzeniu całej populacji, zameniamy starą populację na nową
        self.population = new_population
        self.fitness = new_fitness

        best = np.min(self.fitness)
        average = np.mean(self.fitness)
        worst = np.max(self.fitness)
        return best, average, worst, np.copy(self.population)
