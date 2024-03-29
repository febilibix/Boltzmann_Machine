from BM import BoltzmannMachine
import numpy as np
import matplotlib.pyplot as plt

import os

def rand_sal_data(n_neurons, data_path, n_patterns, seed=0):
    np.random.seed(0)
    rand_neurons = np.random.randint(160, size=n_neurons)
    sal_data = np.loadtxt(data_path)[rand_neurons, :n_patterns]
    sal_data[np.where(sal_data == 0)] = -1 # change 0 to -1
    return sal_data

def main():
    # set params
    n = 20 # number of spins
    p = 953
    eta = .005 # learning rate
    epsilon = 1e-13 # convergence criterion
    num_iter = 1000 # number of MH steps

    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/bint.txt"

    # data = np.random.choice([-1, 1], size=(n,p))
    data = rand_sal_data(n, data_path, p)
    print(data.shape)

    # sal_data = rand_sal_data(n, data_path, p)
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2)
    # axes[0].imshow(sal_data)
    # axes[1].imshow(data)
    # plt.show()
    # quit()


    w_init = np.ones((n,n))
    w = w_init = w_init - np.diag(w_init)
    theta = theta_init = np.random.randn(n)

    fp_epsilons = 10.**(np.arange(1, -2, -.2))
    LLs = []
    

    for eps in fp_epsilons:
        bm = BoltzmannMachine(data, w, theta, eta, epsilon, num_iter)
        LLs.append(bm.solve_fixed_points(eps))
        print("Log likelihood for ")

    plt.plot(fp_epsilons, LLs)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Log Likelihood")
    plt.title(r"Likelihood for different $\epsilon$ for salamander data" + "\n neurons = 20")
    plt.grid()
    cur_path = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.dirname(cur_path) + "/plots/"
    plt.savefig(plot_path + "ex6")

if __name__ == "__main__":
    main()