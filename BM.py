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
        self.all_LLs = dict(exact = list(), MH = list()) 

        self.get_clamped_stats() ## DUNNO if at some point this comes out of the init()

        self.methods = {"exact": self.train_exact, "MH": self.train_MH}


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


    def log_likelihood(self, x):
        prob = self.boltzmann_gibbs_normalized(x)
        if np.isscalar(prob):
            prob = np.array([prob])
        return np.einsum("i->",np.log(prob))/self.P
    

    def train_exact(self):

        change = 1
        while np.abs(change) > self.epsilon: 
            all_probs = self.boltzmann_gibbs_normalized(self.all_configurations)
            assert(np.isclose(np.sum(all_probs),1))

            mu = self.all_configurations @ all_probs
            Sigma = np.einsum('ik,k,jk->ij',self.all_configurations, all_probs, self.all_configurations) 

            dw = (self.Sigma_c - Sigma)
            dtheta = (self.mu_c - mu)

            self.w += self.eta * dw
            self.theta += self.eta * dtheta

            change = np.max((np.max(np.abs(dw)), np.max(np.abs(dtheta))))
            self.all_LLs["exact"].append(self.log_likelihood(self.data))
            print(change)

        print("Converged.")


    def spin_flip(self, state, flip_loc):
        x = state.copy()
        x[flip_loc] *= -1
        return x
    

    # calc energy difference of spin flip for BM
    def energy_diff(self, x, flip_loc):
        return (2*x[flip_loc]*self.w[flip_loc,:]@x + 2*x[flip_loc]*self.theta[flip_loc])

    # def mh(x_init, w, t, n_samples):
    #     samples = np.zeros((int(n_samples), len(x_init)))
    #     samples[0,:]=x_init
    #     for i in range(int(n_samples)-1):
    #         flip_loc = np.random.randint(len(x_init))
    #         a = min(1, np.exp(energy_diff(flip_loc, samples[i,:], w, t)))
    #         if np.random.uniform(0,1)<a:
    #             samples[i+1,:] = spin_flip(x_init, flip_loc)
    #         else:
    #             samples[i+1, :] = samples[i, :]
    #     return samples


    def metropolis_hastings(self, num_samples):
        initial_state = np.random.randn(self.N)
        samples = [initial_state]

        acceptance_ratio=np.empty(num_samples)
        accepted = []

        for i in range(num_samples-1):
            current_state = samples[-1]

            # Sample random spin flip location
            flip_loc = np.random.randint(0, self.N - 1)

            # Propose a new state from the proposal distribution

            # Calculate the likelihood for the proposed and current states
            # likelihood_current = self.boltzmann_gibbs_normalized(current_state)
            # likelihood_proposed = self.boltzmann_gibbs_normalized(current_state[flip_loc] *= -1)

            # Calculate acceptance ratio
            acceptance_ratio[i] = min(1, np.exp(-self.energy_diff(current_state, flip_loc)))

            # Accept or reject the proposed state
            if np.random.uniform(0, 1) < acceptance_ratio[i]:
                proposed_state = self.spin_flip(current_state, flip_loc)
                samples.append(proposed_state)
                accepted.append(1)
            else:
                samples.append(current_state)
                accepted.append(0)

        acceptance_ratio[-1] = 1

        samples = np.array(samples).T
        mu = np.mean(samples, axis = 1)
        Sigma = np.einsum("ik, jk -> ij", samples, samples)/num_samples

        return mu, Sigma #, current_state


    def train_MH(self, num_samples = 1000):

        change = 1
        last_changes = []
        # mu = np.random.randn(self.N)

        # while np.abs(change) > self.epsilon:
        for _ in range(100):
            all_probs = self.boltzmann_gibbs_normalized(self.all_configurations)

            # assert(np.isclose(np.sum(all_probs),1))
            print(np.sum(all_probs))

            mu, Sigma = self.metropolis_hastings(num_samples)

            dw = (self.Sigma_c - Sigma)
            dtheta = (self.mu_c - mu)

            self.w += self.eta * dw 
            self.theta += self.eta * dtheta 

            change_current = np.max((np.max(np.abs(dw)), np.max(np.abs(dtheta))))

            last_changes.append(change_current)
            if len(last_changes) > 100:
                last_changes.pop(0)

            print(change_current)
            
            change = np.mean(last_changes)

            self.all_LLs["MH"].append(self.log_likelihood(self.data))
            print(change)

        print("Converged")



    def plot_LL(self, method = "exact"):

        if len(self.all_LLs[method]) == 0:
            train = self.methods[method]
            train()
        #TODO: THIS OF COURSE NEEDS TO BE DONE PROPERLY HAHA

        plt.figure()
        plt.plot(range(len(self.all_LLs[method])), self.all_LLs[method])
        plt.savefig(f"plots/LL_toydata_{method}.png")


def main():
    # set params
    n = 10 # number of spins
    p = 30 # number of patterns
    eta = .3 # learning rate
    epsilon = 1e-13 # convergence criterion

    # generate random data set with 10-20 spins
    x_small = np.random.choice([-1, 1], size=(n,p))
    w_init = np.random.randn(n,n)
    w = w_init = w_init - np.diag(w_init)
    theta = theta_init = np.random.randn(n)

    bm = BoltzmannMachine(x_small, w, theta, eta, epsilon)
    bm.plot_LL("MH")


if __name__ == "__main__":
    main()

