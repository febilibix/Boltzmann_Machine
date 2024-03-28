from BM import BoltzmannMachine
import numpy as np
import os

def rand_sal_data(n_neurons, data_path, seed=0):
    np.random.seed(0)
    rand_neurons = np.random.randint(160, size=n_neurons)
    sal_data = np.loadtxt(data_path)[rand_neurons, :200]
    sal_data[np.where(sal_data == 0)] = -1 # change 0 to -1
    return sal_data

def main():
    # set params
    n = 5 # number of spins
    p = 50 # number of patterns
    eta = .005 # learning rate
    epsilon = 1e-13 # convergence criterion
    num_iter = 1000 # number of MH steps

    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/bint.txt"
    data = rand_sal_data(n, data_path)
    w_init = np.ones((n,n))
    w = w_init = w_init - np.diag(w_init)
    theta = theta_init = np.random.randn(n)

    fp_epsilons = 10.**(np.linspace(-1, -5, 5))
    fp_epsilons = 10.**(np.arange(1, -2, -.2))
    LLs = []
    for eps in fp_epsilons:
        bm = BoltzmannMachine(data, w, theta, eta, epsilon, num_iter)
        bm.solve_fixed_points(eps)
        print("test")

if __name__ == "__main__":
    main()