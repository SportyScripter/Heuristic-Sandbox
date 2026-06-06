import numpy as np
import functions as fn
from algorihtms import EvolutionaryAlgorithm, DifferentialEvolution
from visualization import animate_comparison_3d, plot_convergence


def main():
    dim = 3
    bounds = np.array([[-5.12, 5.12] for _ in range(dim)])
    max_iterations = 100
    population_size = 200
    func_to_test = fn.rastrigin

    ae_mut_rate = 0.2
    ae_mut_scale = 0.5

    de_F = 0.5
    de_CR = 0.1

    print(f"---- Start Wielkiego Porównania: AE vs DE ----")
    print(f"Funkcja: Rastrigin | Wymiar: {dim} | Iteracje: {max_iterations}\n")

    algo_ae = EvolutionaryAlgorithm(
        func=func_to_test,
        bounds=bounds,
        population_size=population_size,
        mutation_rate=ae_mut_rate,
        mutation_scale=ae_mut_scale,
    )

    algo_de = DifferentialEvolution(
        func=func_to_test,
        bounds=bounds,
        population_size=population_size,
        F=de_F,
        CR=de_CR,
    )

    algo_de = DifferentialEvolution(
        func=func_to_test,
        bounds=bounds,
        population_size=population_size,
        F=de_F,
        CR=de_CR,
    )
    hist_ae, hist_de = animate_comparison_3d(algo_ae, algo_de, max_iterations)
    if hist_ae and hist_de:
        plot_convergence(hist_ae, hist_de)


if __name__ == "__main__":
    main()
