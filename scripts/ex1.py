import numpy as np
from BM import BoltzmannMachine
import matplotlib.pyplot as plt
import os 

def main():
    n = 10 # number of spins
    p = 50 # number of patterns
    eta = .005 # learning rate
    epsilon = 1e-13 # convergence criterion
    num_iter = 1000 # number of MH steps

    # generate random data set with 10-20 spins
    x_small = np.random.choice([-1, 1], size=(n,p))
    w_init = np.ones((n,n))
    w = w_init = w_init - np.diag(w_init)
    theta = np.random.randn(n)

    bm = BoltzmannMachine(x_small, w, theta, eta, epsilon, num_iter)
    bm.train("exact")

    return bm

def make_plot(bm, file_path):
    plt.figure()
    plt.plot(range(len(bm.all_LLs[bm.method])), bm.all_LLs[bm.method])
    plt.xlabel('Runs')
    plt.ylabel('Log Likelihood')
    plt.grid()
    plt.title("Exact Solution of Boltzmann Machine \n n_neurons = 10")
    plt.savefig(file_path)



if __name__ == "__main__":
    trained_bm = main()

    cur_path = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.dirname(cur_path) + "/plots/"
    make_plot(trained_bm, plot_path + "ex1")

