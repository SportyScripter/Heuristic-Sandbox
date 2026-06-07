import numpy as np
import functions as fn
from algorithm import EvolutionaryAlgorithm, DifferentialEvolution
from visualization import animate_comparison_3d, plot_convergence


def main():
    # --- PARAMETRY ŚRODOWISKA ---
    # UWAGA: Jeśli dim == 3, włączy się animacja 3D. Dla innych wartości zadziała tylko tryb tekstowy.
    dim = 3
    bounds = np.array([[-5.12, 5.12] for _ in range(dim)])
    max_iterations = 100
    population_size = 200

    # Wybór z 5 dostępnych funkcji (np. rastrigin, ackley, rosenbrock, sphere, griewank)
    func_to_test = fn.rastrigin

    # --- PARAMETRY DLA KLASYCZNEGO AE ---
    ae_mut_rate = 0.2
    ae_mut_scale = 0.5

    # --- PARAMETRY DLA EWOLUCJI RÓŻNICOWEJ (DE) ---
    de_F = 0.5
    de_CR = 0.1

    print("---- Start Porównania: AE vs DE ----")
    print(
        f"Funkcja: {func_to_test.__name__} | Wymiary: {dim}D | Pokolenia: {max_iterations}"
    )
    print(f"Populacja: {population_size} osobników\n")

    # --- INICJALIZACJA SILNIKÓW OPTYMALIZACYJNYCH ---
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

    # --- URUCHOMIENIE SYMULACJI ---
    hist_ae = []
    hist_de = []

    if dim == 3:
        # Wizualizacja położenia osobników dla funkcji 3-wymiarowej
        hist_ae, hist_de = animate_comparison_3d(algo_ae, algo_de, max_iterations)
    else:
        # Dla wymiarów > 3 rysowanie 3D jest niemożliwe.
        # Uruchamiamy pętlę i raportujemy wyniki w konsoli.
        print(
            "[ Wymiar inny niż 3D - pomijam animację. Uruchamiam pełną moc obliczeniową... ]\n"
        )
        print(
            f"{'ITER':<6} | {'AE BEST':<10} {'AE AVG':<10} {'AE WORST':<10} | {'DE BEST':<10} {'DE AVG':<10} {'DE WORST':<10}"
        )
        print("-" * 80)

        for i in range(max_iterations):
            # Wykonanie kroku i pobranie statystyk z obu algorytmów
            best_ae, avg_ae, worst_ae, _ = algo_ae.step()
            best_de, avg_de, worst_de, _ = algo_de.step()

            hist_ae.append(best_ae)
            hist_de.append(best_de)

            # Raportowanie wyników (najlepsze, średnie, najgorsze) w konsoli (co 10 iteracji i na końcu)
            if i % 10 == 0 or i == max_iterations - 1:
                print(
                    f"{i:<6} | {best_ae:<10.4f} {avg_ae:<10.4f} {worst_ae:<10.4f} | {best_de:<10.4f} {avg_de:<10.4f} {worst_de:<10.4f}"
                )

    # --- RAPORTOWANIE KOŃCOWE (Wykres 2D) ---
    if hist_ae and hist_de:
        plot_convergence(hist_ae, hist_de)


if __name__ == "__main__":
    main()
