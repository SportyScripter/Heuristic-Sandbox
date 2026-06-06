from tsp_data import load_tsplib_data, DATASETS
from tsp_algorithm import TSPEvolutionaryAlgorithm
from tsp_visualization import animate_tsp_step


def main():
    print("--- Problem Komiwojażera (TSP) z TSPlib ---")
    print("Dostępne zestawy:", DATASETS)

    chosen_dataset = "kroA100"

    coords, dist_matrix = load_tsplib_data(chosen_dataset)
    print(f"Pomyślnie załadowano {len(coords)} miast.")

    algo = TSPEvolutionaryAlgorithm(
        dist_matrix=dist_matrix,
        pop_size=100,
        mutation_rate=0.5,
    )

    max_iterations = 2500

    animate_tsp_step(algo, coords, chosen_dataset, max_iterations)


if __name__ == "__main__":
    main()
