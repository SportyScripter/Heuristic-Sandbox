import matplotlib.pyplot as plt


def animate_tsp_step(algo, coords, dataset_name, optimal_val, max_iterations):
    """
    Rysuje na żywo, jak algorytm próbuje rozplątać trasę między miastami.
    """
    # Włączamy tryb interaktywny, żeby wykres mógł się odświeżać w pętli
    # bez zawieszania całego programu.
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))

    print(f"\n[ Rozpoczynam symulację szukania trasy dla: {dataset_name} ]")

    for i in range(max_iterations):
        best_dist, avg_dist, worst_dist, best_route = algo.step()

        # Rysujemy co 5. krok. To jest kompromis między płynnością animacji a czasem jej trwania.
        if i % 5 == 0 or i == max_iterations - 1:
            ax.clear()

            # Układamy współrzędne miast w takiej kolejności, jak idzie nasza najlepsza trasa
            ordered_coords = coords[best_route]

            # Dopina koniec trasy z powrotem do pierwszego miasta,
            ordered_coords = list(ordered_coords)
            ordered_coords.append(ordered_coords[0])

            # Rozdzielamy na listy osi X i Y do narysowania
            xs = [c[0] for c in ordered_coords]
            ys = [c[1] for c in ordered_coords]

            # Rysowanie samej mapy. Trasa jest niebieska, miasta czerwone. Rozmiar kropek wynosi 15.
            ax.plot(xs, ys, linestyle="-", color="blue", alpha=0.6, zorder=1)
            ax.scatter(xs, ys, color="red", s=15, zorder=2)

            # Wyróżniamy miasto startowe dużą, zieloną gwiazdką
            ax.scatter(
                xs[0],
                ys[0],
                color="lime",
                s=150,
                marker="*",
                zorder=3,
                edgecolors="black",
                label="Start",
            )

            # Liczymy (w procentach), jak bardzo nasza trasa różni się od optimum
            gap_percent = ((best_dist - optimal_val) / optimal_val) * 100

            title = (
                f"Zbiór: {dataset_name} (Miast: {algo.num_cities}) | Iteracja: {i}\n"
                f"Obecna trasa: {best_dist:.0f} | OPTIMUM: {optimal_val}\n"
                f"Strata do optymalnego rozwiązania: {gap_percent:.2f}%"
            )

            ax.set_title(title, fontweight="bold")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)

            # Odrysowujemy i robimy pauzę, żeby system zdążył wyświetlić klatkę
            plt.draw()
            plt.pause(0.01)

    # Wyłączamy tryb interaktywny po zakończeniu pętli, żeby okno z wykresem
    # nie zniknęło nam samo, tylko czekało na ręczne zamknięcie.
    plt.ioff()
    print("[ Zakończono symulację. Zamknij okno wykresu. ]")
    plt.show()
