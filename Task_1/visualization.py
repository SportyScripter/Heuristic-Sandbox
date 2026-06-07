import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# WIZUALIZACJA 3D
# =====================================================================


def animate_comparison_3d(algo_ae, algo_de, max_iterations):
    """
    Rysuje na żywo dwa obok siebie wykresy 3D.
    Pokazuje, jak populacja z AE i populacja z DE szukają zera.
    """
    # Włączamy tryb interaktywny, żeby wykres odświeżał się sam w pętli
    plt.ion()
    fig = plt.figure(figsize=(14, 7))

    # Ustawiamy dwa osobne wykresy na jednym ekranie
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    print("\n[ Rozpoczynam symulację krokową 3D: AE vs DE... ]")

    # Listy do zapisywania najlepszego wyniku z każdej iteracji (potrzebne do wykresu 2D na koniec)
    history_ae = []
    history_de = []

    for i in range(max_iterations):
        # Oba algorytmy wykonują jeden krok (jedną generację)
        best_ae, _, _, pop_ae = algo_ae.step()
        best_de, _, _, pop_de = algo_de.step()

        # Zapisujemy najlepsze dotąd wyniki z tego kroku do historii
        history_ae.append(best_ae)
        history_de.append(best_de)

        # Co iteracje czyścimy wykres i rysujemy od nowa, żeby pokazać aktualną populację
        ax1.clear()
        ax2.clear()

        # --- Rysujemy lewy wykres dla Klasycznego AE ---
        # Używamy mapy kolorów 'coolwarm_r', żeby kolorem kropki (zmienna c)
        # pokazywać jakość osobnika (jego fitness)
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
        # Ustawiamy na sztywno ramy "pudełka", w którym szukamy, żeby
        # kamera nie skakała po całym ekranie, gdy osobniki się przemieszczają.
        ax1.set_xlim(algo_ae.bounds[0, 0], algo_ae.bounds[0, 1])
        ax1.set_ylim(algo_ae.bounds[1, 0], algo_ae.bounds[1, 1])
        ax1.set_zlim(algo_ae.bounds[2, 0], algo_ae.bounds[2, 1])

        # --- Rysujemy prawy wykres dla Ewolucji Różnicowej (DE) ---
        # Używamy mapy 'plasma', żeby łatwiej było je od siebie odróżnić
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

        # Rysujemy odświeżony wykres z przerwą, żeby animacja była płynna
        plt.draw()
        plt.pause(0.01)

    # Wyłączamy interaktywność i czekamy aż użytkownik kliknie X
    plt.ioff()
    print(
        "[ Zakończono symulację 3D. Zamknij okno wykresu, aby zobaczyć wykres porównawczy 2D. ]"
    )
    plt.show()

    return history_ae, history_de


# =====================================================================
# WIZUALIZACJA 2D (Statyczny wykres na podsumowanie)
# =====================================================================


def plot_convergence(history_ae, history_de):
    """
    Rysuje wykres liniowy pokazujący, który algorytm szybciej i głębiej wpadł w dolinę.
    Użycie skali logarytmicznej na osi Y.
    """
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

    # Ponieważ wartości w pierwszych iteracjach są wielkie a w późniejszych bardzo małe,
    # skala logarytmiczna pozwala zobaczyć każdy krok.
    plt.yscale("log")

    plt.title("Krzywa zbieżności: AE vs DE", fontsize=14, fontweight="bold")
    plt.xlabel("Iteracja (Pokolenie)", fontsize=12)
    plt.ylabel("Najlepsza wartość funkcji celu (skala logarytmiczna)", fontsize=12)

    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    print("\n[ Generowanie wykresu zbieżności 2D... ]")
    plt.show()
