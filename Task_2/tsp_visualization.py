import matplotlib.pyplot as plt

def animate_tsp_step(algo, coords, dataset_name, max_iterations):
    """Animuje proces znajdowania najkrótszej trasy krok po kroku."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    print(f"\n[ Rozpoczynam symulację szukania trasy dla: {dataset_name} ]")

    for i in range(max_iterations):
        best_dist, avg_dist, worst_dist, best_route = algo.step()

        ax.clear()
        
        # Pobieramy ułożone współrzędne według aktualnie najlepszej trasy
        ordered_coords = coords[best_route]
        # Dodajemy pierwsze miasto na koniec, żeby zamknąć pętlę na rysunku
        ordered_coords = list(ordered_coords)
        ordered_coords.append(ordered_coords[0])
        
        xs = [c[0] for c in ordered_coords]
        ys = [c[1] for c in ordered_coords]

        # Rysujemy linie (trasę) i punkty (miasta)
        ax.plot(xs, ys, linestyle='-', color='blue', alpha=0.7, zorder=1)
        ax.scatter(xs, ys, color='red', s=40, zorder=2, edgecolors='black')
        
        # Zaznaczamy start zieloną gwiazdką
        ax.scatter(xs[0], ys[0], color='lime', s=200, marker='*', zorder=3, edgecolors='black', label='Start')

        ax.set_title(f"Dataset: {dataset_name} | Iteracja: {i}\nNajkrótsza znaleziona trasa: {best_dist:.2f}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    print("[ Zakończono symulację. Zamknij okno wykresu. ]")
    plt.show()