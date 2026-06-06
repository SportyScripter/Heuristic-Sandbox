import numpy as np

# --- 4 WYMAGANE OPERATORY ZMIANY TRASY ---


def mutate_swap(route):
    """Zamienia miejscami dwa losowe miasta."""
    idx = np.random.choice(len(route), 2, replace=False)
    route[idx[0]], route[idx[1]] = route[idx[1]], route[idx[0]]
    return route


def mutate_reverse(route):
    """Odwraca kolejność losowego fragmentu trasy (często najlepsza mutacja dla TSP!)."""
    i, j = sorted(np.random.choice(len(route), 2, replace=False))
    route[i : j + 1] = route[i : j + 1][::-1]
    return route


def mutate_insert(route):
    """Zabiera miasto z jednej pozycji i wstawia w inną."""
    i, j = np.random.choice(len(route), 2, replace=False)
    city = route.pop(i)
    route.insert(j, city)
    return route


def mutate_scramble(route):
    """Losowo miesza miasta na wybranym fragmencie trasy."""
    i, j = sorted(np.random.choice(len(route), 2, replace=False))
    sub_route = route[i : j + 1]
    np.random.shuffle(sub_route)
    route[i : j + 1] = sub_route
    return route


# --- ALGORYTM EWOLUCYJNY DLA TSP ---


class TSPEvolutionaryAlgorithm:
    def __init__(self, dist_matrix, pop_size=100, mutation_rate=0.3):
        self.dist_matrix = dist_matrix
        self.num_cities = len(dist_matrix)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        # Lista 4 operatorów, algorytm będzie losował, którego użyć
        self.mutations = [mutate_swap, mutate_reverse, mutate_insert, mutate_scramble]

        # Inicjalizacja: każda trasa to losowa permutacja liczb od 0 do N-1
        self.population = [
            list(np.random.permutation(self.num_cities)) for _ in range(pop_size)
        ]
        self.fitness = np.array(
            [self.calculate_distance(ind) for ind in self.population]
        )

    def calculate_distance(self, route):
        """Oblicza całkowitą długość trasy w kilometrach (wraz z powrotem do startu)."""
        dist = 0.0
        for i in range(self.num_cities - 1):
            dist += self.dist_matrix[route[i], route[i + 1]]
        # Powrót z ostatniego miasta do pierwszego
        dist += self.dist_matrix[route[-1], route[0]]
        return dist

    def step(self):
        """Krok algorytmu dla TSP (Z selekcją turniejową i czystą mutacją)."""
        new_population = []

        # Elitaryzm - najlepszy z poprzedniej epoki przechodzi dalej za darmo (ratuje przed "zębami piły")
        best_idx = np.argmin(self.fitness)
        new_population.append(self.population[best_idx].copy())

        while len(new_population) < self.pop_size:
            # 1. Selekcja turniejowa (wybieramy lepszego z 3 losowych)
            tournament = np.random.choice(self.pop_size, 3, replace=False)
            winner_idx = tournament[np.argmin(self.fitness[tournament])]
            parent = self.population[winner_idx].copy()

            # 2. Mutacja (Krzyżowanie w TSP jest trudne, więc polegamy silnie na naszych 4 mutacjach)
            if np.random.rand() < self.mutation_rate:
                # Wybieramy losowo 1 z 4 operatorów!
                mutation_func = np.random.choice(self.mutations)
                parent = mutation_func(parent)

            new_population.append(parent)

        self.population = new_population
        self.fitness = np.array(
            [self.calculate_distance(ind) for ind in self.population]
        )

        best = np.min(self.fitness)
        average = np.mean(self.fitness)
        worst = np.max(self.fitness)

        best_route = self.population[np.argmin(self.fitness)]
        return best, average, worst, best_route

#dopisanie temperatury do algorytmu TSP (choć to już bardziej hybryda z symulowanym wyżarzaniem, ale może być ciekawym eksperymentem!)