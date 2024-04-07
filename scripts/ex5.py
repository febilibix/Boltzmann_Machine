import numpy as np
from BM import BoltzmannMachine
import matplotlib.pyplot as plt
import os 

def make_plot(bm, name):
    len_x = 5000
    plt.plot(range(0, len_x), bm.all_LLs[bm.method][0:len_x], label=name)

def main(method, num_iter, num_iter_MF=1e6):
    n = 15 # number of spins
    p = 50 # number of patterns
    eta = .005 # learning rate
    epsilon = 1e-7 # convergence criterion

    # generate random data set with 10-20 spins
    x_small = np.random.choice([-1, 1], size=(n,p))
    w_init = np.ones((n,n))
    w = w_init = w_init - np.diag(w_init)
    theta = np.random.randn(n)

    bm = BoltzmannMachine(x_small, w, theta, eta, epsilon, num_iter)
    bm.train(method, max_iter_MF=num_iter_MF)

    return bm


if __name__ == "__main__":
    np.random.seed(0)
    # exact_bm = main('exact', 0)
    MF_bm_5 = main('MF', 0, 50)
    MF_bm_100 = main('MF', 0, 100)
    MF_bm_500 = main('MF', 0, 500)
    MF_bm_1000 = main('MF', 0, 1000)
    MF_bm_10000 = main('MF', 0, 10000)

    plt.figure()
    # make_plot(exact_bm, "Exact")
    make_plot(MF_bm_5, "MF iter = 50")
    make_plot(MF_bm_100, "MF iter = 100")
    make_plot(MF_bm_500, "MF iter = 500")
    make_plot(MF_bm_1000, "MF iter = 1000")
    # TAKES A LOT OF TIME SO 1000 IS GOOD
    # make_plot(MF_bm_10000, "MF iter = 10000")


    plt.legend()
    plt.xlabel('Runs')
    plt.ylabel('Log Likelihood')
    plt.grid()
    plt.title(f"Comparing MF and Exact \n n_neurons = {exact_bm.N}")
    cur_path = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.dirname(cur_path) + "/plots/"
    plt.savefig(plot_path + "ex3_MF_mult")
    plt.show()


