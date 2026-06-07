from tsp_data import load_tsplib_data, TSPLIB_OPTIMALS, DATASETS
from tsp_algorithm import TSPEvolutionaryAlgorithm
from tsp_visualization import animate_tsp_step


def main():
    print("--- Problem Komiwojażera (TSP) > 300 miast ---")
    print("Dostępne zestawy:", DATASETS)

    # Wybór zbioru danych do analizy z naszej listy
    chosen_dataset = "lin318"

    # Pobranie docelowej, optymalnej wartości z naszego słownika.
    optimal_val = TSPLIB_OPTIMALS[chosen_dataset]

    # Wczytanie współrzędnych i gotowej macierzy odległości
    coords, dist_matrix = load_tsplib_data(chosen_dataset)
    print(f"Pomyślnie załadowano {len(coords)} miast.")

    # Inicjalizacja algorytmu ewolucyjnego z symulowanym wyżarzaniem.
    # Ustawiamy mocniejsze parametry, ponieważ przeszukanie ponad 300 miast
    # wymaga znacznie większych zasobów niż mniejsze zbiory danych.
    algo = TSPEvolutionaryAlgorithm(
        dist_matrix=dist_matrix,
        pop_size=300,  # Znacznie większa populacja, żeby zapewnić różnorodność tras
        mutation_rate=0.8,  # Bardzo wysoka mutacja, algorytm ma mocno mieszać w genach
        initial_temp=1500.0,  # Wysoka temperatura startowa, aby na początku pozwalać na ucieczkę z lokalnych minimów
        cooling_rate=0.99,  # Dość wolne chłodzenie (temperatura spada o 1% co krok)
    )

    # Ustawiamy liczbę iteracji na 3000, co jest rozsądnym kompromisem między czasem wykonania a szansą na znalezienie optymalnej trasy.
    max_iterations = 3000

    # Uruchomienie głównej pętli wraz z animacją na żywo
    animate_tsp_step(algo, coords, chosen_dataset, optimal_val, max_iterations)


if __name__ == "__main__":
    main()
