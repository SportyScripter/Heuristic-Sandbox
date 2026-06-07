import numpy as np

# =====================================================================
# FUNKCJE TESTOWE DO OPTYMALIZACJI
# =====================================================================


def sphere(x):
    return np.sum(x**2)


def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley(x):
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))

    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)

    # Dodajemy liczbę Eulera (np.e) i 20, żeby wyzerować najniższy punkt
    return term1 + term2 + 20 + np.e


def griewank(x):
    sum_sq = np.sum(x**2) / 4000
    indices = np.arange(1, len(x) + 1)

    prod_cos = np.prod(np.cos(x / np.sqrt(indices)))

    return sum_sq - prod_cos + 1
