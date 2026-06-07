# Evolution-vs-Differential# Metaheuristics & Optimization Engines 🧬🛤️

This repository contains Python implementations of advanced metaheuristic algorithms designed to solve complex continuous optimization problems and NP-hard combinatorial problems (Traveling Salesperson Problem). 

The project is divided into two main modules, each demonstrating a different application of evolutionary computation.

## 📂 Project Structure

### `01_AE_vs_DE_Comparison`
A comprehensive benchmarking suite comparing a **Classic Evolutionary Algorithm (EA)** with **Differential Evolution (DE)** across n-dimensional continuous spaces.
* **Supported Benchmarks:** Rastrigin, Ackley, Rosenbrock, Sphere, Griewank.
* **N-Dimensional Support:** Robust logic handling $3 \le n < 50$ dimensions. 
* **Visualizations:** Real-time 3D scatter plots for 3D functions, and logarithmic 2D convergence charts to track micro-improvements near global optima.
* **Key Mechanics:** Elitism, tournament selection, Gaussian mutation, and adaptive crossover.

### `02_TSP_Hybrid_Solver`
A highly optimized, hybrid (memetic) solver for the **Traveling Salesperson Problem (TSP)** capable of routing maps with >300 nodes.
* **TSPlib Integration:** Automated fetching and parsing of official large-scale datasets (e.g., `lin318`, `rd400`).
* **Hybrid Architecture:** Combines Evolutionary computation with **Simulated Annealing (SA)** to escape local minima.
* **Smart Initialization (Golden Start):** Seeds the initial population using a Nearest Neighbor greedy approach and filters random routes against the statistical average.
* **Advanced Operators:** Implements specialized routing mutations (Reverse/Inversion, Displacement, Inversion-Insert, Scramble, Swap).
* **Live Analytics:** Renders real-time route optimization paths and calculates the exact Optimality Gap (%) against mathematically proven TSPlib benchmarks.

## 🚀 Installation & Usage

**Prerequisites:** Python 3.8+ 

1. Clone the repository:
   ```bash
   git clone https://github.com/SportyScripter/Heuristic-Sandbox.git
   cd Heuristic-Sandbox

2. Install required dependencies:
    ```bash
    pip install numpy matplotlib

3. Run the modules:
    ```bash
    Task 1: python 01_AE_vs_DE_Comparison/main.py
    Task 2: python 02_TSP_Hybrid_Solver/main_tsp.py