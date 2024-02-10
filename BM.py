import numpy as np 
from itertools import product
import matplotlib.pyplot as plt

np.random.seed(101)

class BoltzmannMachine():

    def __init__(self, data, w, theta, eta, epsilon):
        self.data, self.w, self.theta = data, w, theta
        self.eta, self.epsilon = eta, epsilon
        all_configurations = list(product([-1, 1], repeat=w.shape[0]))
        self.all_configurations = np.array(all_configurations).T
        self.N, self.P = self.data.shape[0], self.data.shape[1]
        self.all_LLs = []

        self.get_clamped_stats() ## DUNNO if at some point this comes out of the init()


    def get_clamped_stats(self):
        self.mu_c = np.mean(self.data, axis = 1)
        self.Sigma_c = np.einsum("ik, jk -> ij", self.data, self.data)/self.P


    def boltzmann_gibbs(self, x):
        w, theta = self.w, self.theta

        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)

        unnormalized = np.exp(.5*np.einsum('ij,ik,jk->k', w, x, x) + np.einsum('i,ik->k', theta, x))

        return unnormalized


    def boltzmann_gibbs_normalized(self, x):
        all_configurations = self.all_configurations

        unnormalized = self.boltzmann_gibbs(x)
        Z = np.sum(self.boltzmann_gibbs(all_configurations))

        out = unnormalized/Z
        if len(out) == 1:
            out = out[0]
        return out


    def log_likelihood(self):
        prob = self.boltzmann_gibbs_normalized(self.data)
        return np.einsum("i->",np.log(prob))/self.P
    

    def train_exact(self):
        all_configurations, w, theta = self. all_configurations, self.w, self.theta

        change = 1
        while np.abs(change) > self.epsilon: 
            all_probs = self.boltzmann_gibbs_normalized(all_configurations)
            print(np.sum(all_probs))
            assert(np.isclose(np.sum(all_probs),1))

            mu = all_configurations @ all_probs
            Sigma = np.einsum('ik,k,jk->ij', all_configurations, all_probs, all_configurations) 

            dw = (self.Sigma_c - Sigma)
            dtheta = (self.mu_c - mu)

            w += self.eta * dw
            theta += self.eta * dtheta

            change = np.max((np.max(np.abs(dw)), np.max(np.abs(dtheta))))
            self.all_LLs.append(self.log_likelihood())
            print(change)

        print("Converged.")


    def plot_LL(self):

        if len(self.all_LLs) == 0:
            self.train_exact()
        #TODO: THIS OF COURSE NEEDS TO BE DONE PROPERLY HAHA

        plt.figure()
        plt.plot(range(len(self.all_LLs)), self.all_LLs)
        plt.savefig("plots/LL_toydata_exact.png")


def main():
    # set params
    n = 10 # number of spins
    p = 30 # number of patterns
    eta = .3 # learning rate
    epsilon = 1e-13 # convergence criterion

    # gemerate random data set with 10-20 spins
    x_small = np.random.choice([-1, 1], size=(n,p))
    w_init = np.random.randn(n,n)
    w = w_init = w_init - np.diag(w_init)
    theta = theta_init = np.random.randn(n)

    bm = BoltzmannMachine(x_small, w, theta, eta, epsilon)
    bm.plot_LL()


if __name__ == "__main__":
    main()

