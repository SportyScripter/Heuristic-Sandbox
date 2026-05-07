import numpy as np
import functions as fn
from algorihtms import EvolutinaryAlgorithm, DifferentialEvolution
from visualization import animate_comparison_3d, plot_convergence


def main():
    dim = 3
    bounds = np.array([[-5.0, 5.0] for _ in range(dim)])
    max_iterations = 100
    print(f"---- Start Optymalizacji: Algorytm Ewolucyjny (AE) ----")
    print(f"Funkcja: Sferyczna | Wymiar: {dim} | Iteracje: {max_iterations}\n")
    func_to_test = fn.griewank
    algo_ae = EvolutinaryAlgorithm(
        func=func_to_test,
        bounds=bounds,
        population_size=50,
        mutation_rate=0.5,
        mutation_scale=0.4,
    )

    algo_de = DifferentialEvolution(
        func=func_to_test, bounds=bounds, population_size=50, F=0.5, CR=0.1
    )
    hist_ae, hist_de = animate_comparison_3d(algo_ae, algo_de, max_iterations)
    if hist_ae and hist_de:
        plot_convergence(hist_ae, hist_de)


if __name__ == "__main__":
    main()
