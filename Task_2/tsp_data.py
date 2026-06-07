import os
import urllib.request
import numpy as np

TSPLIB_OPTIMALS = {
    "lin318": 42029,  # Mapa z 318 miastami
    "rd400": 15281,  # Mapa z 400 miastami
    "fl417": 11861,  # Mapa z 417 miastami
    "pr439": 107217,  # Mapa z 439 miastami
    "pcb442": 50778,  # Mapa z 442 miastami
}

# Robimy z tego listę dostępnych opcji do wyboru w mainie
DATASETS = list(TSPLIB_OPTIMALS.keys())


def download_tsplib(name, folder="data"):
    """
    Funkcja do pobierania plików .tsp z internetu, jak jeszcze ich nie mamy na dysku.
    Zapisuje je w folderze 'data'.
    """
    # Jak nie ma folderu 'data', to go tworzymy
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(current_dir, "data")

    # Tworzymy folder, jeśli jeszcze nie istnieje
    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = os.path.join(folder, f"{name}.tsp")

    # Pobieramy plik tylko wtedy, kiedy go jeszcze nie ma
    if not os.path.exists(filepath):
        print(f"Pobieranie pliku {name}.tsp z internetu...")

        # Używam linku z GitHuba, bo na oficjalnej stronie TSPlib pliki są spakowane w formacie .gz
        url_mirror = (
            f"https://raw.githubusercontent.com/Mastqe/tsplib/master/{name}.tsp"
        )
        try:
            urllib.request.urlretrieve(url_mirror, filepath)
        except Exception as e:
            print(f"Błąd pobierania: {e}")

    return filepath


def load_tsplib_data(name):
    """
    Czyta plik tekstowy, wyciąga z niego koordynaty miast i od razu
    liczy macierz odległości (kto ma do kogo jak daleko).
    """
    filepath = download_tsplib(name)
    coords = []

    # Otwieramy plik i czytamy go linijka po linijce
    with open(filepath, "r") as f:
        lines = f.readlines()
        reading_nodes = False

        for line in lines:
            # Jak trafimy na ten napis, to znaczy, że od następnej linijki lecą współrzędne
            if "NODE_COORD_SECTION" in line:
                reading_nodes = True
                continue

            # EOF oznacza koniec pliku / danych
            if "EOF" in line or line.strip() == "":
                if reading_nodes:
                    break

            # Zbieranie współrzędnych X i Y
            if reading_nodes:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # parts[0] to numer miasta, parts[1] to X, parts[2] to Y
                    coords.append([float(parts[1]), float(parts[2])])

    coords = np.array(coords)
    num_cities = len(coords)

    # Liczymy odległości każdy z każdym (żeby algorytm potem w trakcie
    # ewolucji nie musiał tego liczyć milion razy i działał szybciej)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            # Standard TSPlib narzuca, żeby odległość zaokrąglać
            # do pełnych liczb (integerów). Stąd np.round.
            dist_matrix[i, j] = np.round(np.linalg.norm(coords[i] - coords[j]))

    return coords, dist_matrix
