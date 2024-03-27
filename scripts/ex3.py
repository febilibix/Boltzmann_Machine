import numpy as np
from BM import BoltzmannMachine
import matplotlib.pyplot as plt
import os 

def main(method, num_iter):
    n = 10 # number of spins
    p = 50 # number of patterns
    eta = .005 # learning rate
    epsilon = 1e-7 # convergence criterion

    # generate random data set with 10-20 spins
    x_small = np.random.choice([-1, 1], size=(n,p))
    w_init = np.ones((n,n))
    w = w_init = w_init - np.diag(w_init)
    theta = np.random.randn(n)

    bm = BoltzmannMachine(x_small, w, theta, eta, epsilon, num_iter)
    bm.train(method)

    return bm

def make_plot(bm, name):
    plt.plot(range(len(bm.all_LLs[bm.method])), bm.all_LLs[bm.method], label=name)



if __name__ == "__main__":
    exact_bm = main('exact', 0)
    MH_bm_5 = main('MH', 5)
    MH_bm_100 = main('MH', 100)
    MH_bm_500 = main('MH', 500)
    MH_bm_1000 = main('MH', 1000)
    MH_bm_10000 = main('MH', 10000)

    plt.figure()
    make_plot(exact_bm, "Exact")
    make_plot(MH_bm_5, "MH iter = 5")
    make_plot(MH_bm_100, "MH iter = 100")
    make_plot(MH_bm_500, "MH iter = 500")
    make_plot(MH_bm_1000, "MH iter = 1000")
    # TAKES A LOT OF TIME SO 1000 IS GOOD 
    make_plot(MH_bm_10000, "MH iter = 10000")


    plt.legend()
    plt.xlabel('Runs')
    plt.ylabel('Log Likelihood')
    plt.grid()
    plt.title("Comparing MH and Exact \n n_neurons = 10")
    cur_path = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.dirname(cur_path) + "/plots/"
    plt.savefig(plot_path + "ex3")
    plt.show()


