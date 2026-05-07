import matplotlib.pyplot as plt
import numpy as np


def animate_comparison_3d(algo_ae, algo_de, max_iterations):
    if algo_ae.dim != 3 or algo_de.dim != 3:
        print("Błąd: Wizualizacja 3D obsługuje tylko dokładnie 3 wymiary (n=3)!")
        return None, None
    plt.ion()
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    print("\n[ Rozpoczynam symulację krokową 3D: AE vs DE... ]")
    history_ae = []
    history_de = []
    for i in range(max_iterations):
        best_ae, _, _, pop_ae = algo_ae.step()
        best_de, _, _, pop_de = algo_de.step()
        history_ae.append(best_ae)
        history_de.append(best_de)
        ax1.clear()
        ax2.clear()
        ax1.scatter(
            pop_ae[:, 0],
            pop_ae[:, 1],
            pop_ae[:, 2],
            c=algo_ae.fitness,
            cmap="coolwarm_r",
            s=50,
            edgecolors="black",
        )
        ax1.set_title(f"Klasyczny AE\nIteracja: {i} | Wynik: {best_ae:.5f}")
        ax1.set_xlim(algo_ae.bounds[0, 0], algo_ae.bounds[0, 1])
        ax1.set_ylim(algo_ae.bounds[1, 0], algo_ae.bounds[1, 1])
        ax1.set_zlim(algo_ae.bounds[2, 0], algo_ae.bounds[2, 1])
        ax2.scatter(
            pop_de[:, 0],
            pop_de[:, 1],
            pop_de[:, 2],
            c=algo_de.fitness,
            cmap="plasma",
            s=50,
            edgecolors="black",
        )
        ax2.set_title(f"Ewolucja Różnicowa (DE)\nIteracja: {i} | Wynik: {best_de:.5f}")
        ax2.set_xlim(algo_de.bounds[0, 0], algo_de.bounds[0, 1])
        ax2.set_ylim(algo_de.bounds[1, 0], algo_de.bounds[1, 1])
        ax2.set_zlim(algo_de.bounds[2, 0], algo_de.bounds[2, 1])
        plt.draw()
        plt.pause(0.1)
    plt.ioff()
    print(
        "[ Zakończono symulację 3D. Zamknij okno wykresu, aby zobaczyć wykres porównawczy 2D. ]"
    )
    plt.show()
    return history_ae, history_de


def plot_convergence(history_ae, history_de):
    plt.figure(figsize=(10, 6))
    plt.plot(
        history_ae,
        label="Algorytm Ewolucyjny (AE)",
        linewidth=2.5,
        color="blue",
        alpha=0.8,
    )
    plt.plot(
        history_de,
        label="Ewolucja Różnicowa (DE)",
        linewidth=2.5,
        color="orange",
        alpha=0.9,
    )
    plt.yscale("log")
    plt.title("Krzywa zbieżności: AE vs DE", fontsize=14, fontweight="bold")
    plt.xlabel("Iteracja (Pokolenie)", fontsize=12)
    plt.ylabel("Najlepsza wartość funkcji celu (skala logarytmiczna)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    print("\n[ Generowanie wykresu zbieżności 2D... ]")
    plt.show()
