from BM import BoltzmannMachine
import numpy as np 

def main():
    data = np.loadtxt("data/bint.txt")
    data[data == 0] = -1
    eta = .005 # learning rate
    epsilon = 1e-13 # convergence criterion
    num_iter = 100

    n = data.shape[0]
    p = data.shape[1]

    w_init = np.ones((n,n))
    w = w_init = w_init - np.diag(w_init)
    theta = theta_init = np.random.randn(n)

    # TODO: As is, running this kills my terminal straight away. Even initializing requires 
    # computation of all possible configurations which is intractable. Needs to be changed!
    bm = BoltzmannMachine(data, w, theta, eta, epsilon, num_iter)
    # bm.plot_LL("salamander", "MH")
    
    bm.solve_fixed_points()

if __name__ == "__main__":
    main()