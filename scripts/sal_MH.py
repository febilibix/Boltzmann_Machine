import numpy as np
from BM import BoltzmannMachine

# The original data file has dimension 160 × 283041, which are 297 repeated experiments,
# each of which has 953 time points. Use only one of these repeats for training the BM, 
# ie. your data file for training has dimension n_neurons × 953. 

def rand_sal_data(n_neurons, data_path, seed=0):
    np.random.seed(0)
    rand_neurons = np.random.randint(160, size=n_neurons)
    sal_data = np.loadtxt(data_path)[rand_neurons, :200]
    sal_data[np.where(sal_data == 0)] = -1 # change 0 to -1
    return sal_data

def main():
    # set params
    # TODO: At 30 spins /unnormalized goes to inf and never congerges
    n_spins = 160 # number of spins
    eta = .005 / n_spins # learning rate 
    # TODO does this apply to both MH and MF
    epsilon = 1e-13 # convergence criterion
    num_iter = 1000 # number of MH steps

    # Generate weights and theta
    data = rand_sal_data(n_spins, "data/bint.txt")
    w_init = np.ones((n_spins, n_spins))
    w = w_init = w_init - np.diag(w_init)
    theta = np.random.randn(n_spins)


    bm = BoltzmannMachine(data, w, theta, eta, epsilon, num_iter, approx_partition=True)
    # TODO: change data
    bm.plot_LL("test", "MH")


if __name__ == "__main__":
    main()