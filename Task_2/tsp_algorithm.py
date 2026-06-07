import numpy as np

# --- 6 OPERATORÓW ZMIANY TRASY  ---


def mutate_swap(route):
    """Zamienia miejscami dwa losowe miasta"""
    idx = np.random.choice(len(route), 2, replace=False)
    route[idx[0]], route[idx[1]] = route[idx[1]], route[idx[0]]
    return route


def mutate_reverse(route):
    """Odwraca kolejność fragmentu trasy"""
    i, j = sorted(np.random.choice(len(route), 2, replace=False))
    route[i : j + 1] = route[i : j + 1][::-1]
    return route


def mutate_insert(route):
    """Wyciąga jedno miasto i wkleja je w losowe miejsce"""
    i, j = np.random.choice(len(route), 2, replace=False)
    city = route.pop(i)
    route.insert(j, city)
    return route


def mutate_scramble(route):
    """Bierze fragment trasy i losowo w nim miesza miasta."""
    i, j = sorted(np.random.choice(len(route), 2, replace=False))
    sub_route = route[i : j + 1]
    np.random.shuffle(sub_route)
    route[i : j + 1] = sub_route
    return route


def mutate_displacement(route):
    """Wyciąga cały kawałek trasy i wkleja go w inne miejsce"""
    n = len(route)
    i, j = sorted(np.random.choice(n, 2, replace=False))

    # Zabezpieczenie: jak wylosuje całą trasę, to nic nie robimy
    if i == 0 and j == n - 1:
        return route

    block = route[i : j + 1]
    route = route[:i] + route[j + 1 :]

    # Wklejamy z powrotem w nowe miejsce
    insert_pos = np.random.randint(0, len(route) + 1)
    route = route[:insert_pos] + block + route[insert_pos:]
    return route


def mutate_inversion_insert(route):
    """Wyciąga kawałek trasy, odwraca go i wkleja w inne miejsce."""
    n = len(route)
    i, j = sorted(np.random.choice(n, 2, replace=False))

    if i == 0 and j == n - 1:
        return route[::-1]

    block = route[i : j + 1][::-1]
    route = route[:i] + route[j + 1 :]

    insert_pos = np.random.randint(0, len(route) + 1)
    route = route[:insert_pos] + block + route[insert_pos:]
    return route


# --- ALGORYTM GŁÓWNY I POMOCNICZE ---


def generate_greedy_route(dist_matrix, start_city):
    """
    Algorytm zachłanny. Zaczyna w wybranym mieście i
    zawsze idzie do najbliższego sąsiada.
    """
    num_cities = len(dist_matrix)
    unvisited = set(range(num_cities))

    current_city = start_city
    route = [current_city]
    unvisited.remove(current_city)

    while unvisited:
        # Znajdź najbliższe miasto z tych, w których jeszcze nie byliśmy
        next_city = min(unvisited, key=lambda city: dist_matrix[current_city, city])
        route.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city

    return route


class TSPEvolutionaryAlgorithm:
    def __init__(
        self,
        dist_matrix,
        pop_size=100,
        mutation_rate=0.5,
        initial_temp=1000.0,
        cooling_rate=0.99,
    ):
        self.dist_matrix = dist_matrix
        self.num_cities = len(dist_matrix)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        # Dodatkowe parametry do symulowanego wyżarzania
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate

        self.mutations = [
            mutate_swap,
            mutate_reverse,
            mutate_insert,
            mutate_scramble,
            mutate_displacement,
            mutate_inversion_insert,
        ]

        # Wagi mutacji - najczęściej używamy reverse (35%), bo najlepiej działa
        self.mutation_weights = [0.05, 0.35, 0.10, 0.10, 0.25, 0.15]
        self.population = []

        print("[ Start: Generowanie początkowej populacji... ]")

        # Wrzucamy kilka tras wygenerowanych zachłannie, żeby algorytm
        # miał lepszy punkt startowy niż czysta losowość (0.0 - pełna losowość, tak jakby go nie było).
        num_greedy = int(pop_size * 0.20)
        greedy_starts = np.random.choice(self.num_cities, num_greedy, replace=False)

        for start_node in greedy_starts:
            smart_route = generate_greedy_route(self.dist_matrix, start_node)
            self.population.append(smart_route)

        # Reszta to trasy losowe, ale odrzucamy te najbardziej beznadziejne.
        # Liczymy średnią z 30 losowych tras.
        sample_routes = [
            list(np.random.permutation(self.num_cities)) for _ in range(30)
        ]
        avg_random_dist = np.mean([self.calculate_distance(r) for r in sample_routes])

        while len(self.population) < pop_size:
            route = list(np.random.permutation(self.num_cities))
            dist = self.calculate_distance(route)

            # Jak trasa jest gorsza od średniej, to losujemy ją jeszcze raz
            while dist > avg_random_dist:
                route = list(np.random.permutation(self.num_cities))
                dist = self.calculate_distance(route)

            self.population.append(route)

        # Liczymy długość każdej trasy w populacji
        self.fitness = np.array(
            [self.calculate_distance(ind) for ind in self.population]
        )

    def calculate_distance(self, route):
        """Liczy ile kilometrów ma cała trasa łącznie z powrotem do początku."""
        dist = 0.0
        for i in range(self.num_cities - 1):
            dist += self.dist_matrix[route[i], route[i + 1]]
        dist += self.dist_matrix[route[-1], route[0]]
        return dist

    def step(self):
        """Pojedyncza generacja algorytmu z wyżarzaniem."""
        new_population = []

        # Elitaryzm - najlepsza trasa przechodzi dalej, żeby jej nie zepsuć
        best_idx = np.argmin(self.fitness)
        new_population.append(self.population[best_idx].copy())

        while len(new_population) < self.pop_size:
            # 1. Turniej: losujemy 3 trasy i wybieramy z nich najkrótszą
            tournament = np.random.choice(self.pop_size, 3, replace=False)
            winner_idx = tournament[np.argmin(self.fitness[tournament])]
            parent = self.population[winner_idx].copy()
            parent_fitness = self.fitness[winner_idx]

            # 2. Mutacja wylosowanego rodzica
            if np.random.rand() < self.mutation_rate:
                mutation_func = np.random.choice(
                    self.mutations, p=self.mutation_weights
                )
                child = mutation_func(parent.copy())
                child_fitness = self.calculate_distance(child)

                # --- SYMULOWANE WYŻARZANIE ---
                delta = child_fitness - parent_fitness

                if delta < 0:
                    # Dziecko ma krótszą trasę, bierzemy od razu
                    new_population.append(child)
                else:
                    # Dziecko jest gorsze. Normalnie byśmy je odrzucili, ale
                    # dzięki temperaturze czasami je bierzemy, żeby nie utknąć w miejscu.
                    temp = max(
                        self.temperature, 0.00001
                    )  # zapobiega dzieleniu przez zero
                    acceptance_probability = np.exp(-delta / temp)

                    if np.random.rand() < acceptance_probability:
                        new_population.append(child)
                    else:
                        new_population.append(parent)
            else:
                new_population.append(parent)

        # Zastępujemy starą populację nową
        self.population = new_population
        self.fitness = np.array(
            [self.calculate_distance(ind) for ind in self.population]
        )

        # Zmniejszamy temperaturę (żeby rzadziej akceptować gorsze trasy)
        self.temperature *= self.cooling_rate

        # Statystyki do zwrotu
        best = np.min(self.fitness)
        average = np.mean(self.fitness)
        worst = np.max(self.fitness)
        best_route = self.population[np.argmin(self.fitness)]

        return best, average, worst, best_route
