import os
import urllib.request
import numpy as np

# 5 wybranych zestawów z TSPlib (liczba w nazwie to liczba miast)
DATASETS = ['berlin52', 'eil51', 'st70', 'pr76', 'kroA100']

def download_tsplib(name, folder="data"):
    """Pobiera plik .tsp z serwera, jeśli jeszcze go nie mamy."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = os.path.join(folder, f"{name}.tsp")
    if not os.path.exists(filepath):
        print(f"Pobieranie {name}.tsp z TSPlib...")
        url = f"http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{name}.tsp.gz"
        # TSPlib przechowuje pliki skompresowane (.gz) - dla wygody użyjemy mirrora ze zwykłymi plikami .tsp
        url_mirror = f"https://raw.githubusercontent.com/pdollar/edges/master/edges/data/tsp/{name}.tsp"
        try:
            urllib.request.urlretrieve(url_mirror, filepath)
        except:
            # Awaryjny link do repozytorium z rozpakowanymi plikami TSPlib
            url_backup = f"https://raw.githubusercontent.com/mastqe/tsplib/master/{name}.tsp"
            urllib.request.urlretrieve(url_backup, filepath)
            
    return filepath

def load_tsplib_data(name):
    """Parsuje plik TSPlib i zwraca współrzędne miast oraz macierz odległości."""
    filepath = download_tsplib(name)
    coords = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        reading_nodes = False
        for line in lines:
            if "NODE_COORD_SECTION" in line:
                reading_nodes = True
                continue
            if "EOF" in line:
                break
            if reading_nodes:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # parts[0] to ID, parts[1] to X, parts[2] to Y
                    coords.append([float(parts[1]), float(parts[2])])
                    
    coords = np.array(coords)
    num_cities = len(coords)
    
    # Prekalkulacja macierzy odległości (każdy z każdym) - to drastycznie przyspiesza algorytm!
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            # Odległość euklidesowa (Pitagoras)
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
            
    return coords, dist_matrix