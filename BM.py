import numpy as np 
from itertools import product
import matplotlib.pyplot as plt
from functools import cached_property

np.random.seed(42)

class BoltzmannMachine():

    def __init__(self, data, w, theta, eta, epsilon, num_iter, approx_partition=False):
        self.data, self.w, self.theta = data, w, theta
        self.eta, self.epsilon = eta, epsilon

        self.N, self.P = self.data.shape[0], self.data.shape[1]
        self.approx_partition = approx_partition

        # TODO: Here we will run into a problem sooner or later because we need all configs atm
        # to get log likelihood but for 160 neurons in salamander retina that means 
        # 2**160 = 1461501637330902918203684832716283019655932542976 configurations.
        # I guess we need to approximate Z somehow 

        
        self.all_LLs = dict(exact = list(), MH = list(), MF = list()) 
        self.num_iter = num_iter

        self.get_clamped_stats() ## DUNNO if at some point this comes out of the init()

        # self.mu = np.zeros(self.N)


    def get_clamped_stats(self):
        self.mu_c = np.mean(self.data, axis = 1)
        self.Sigma_c = np.einsum("ik, jk -> ij", self.data, self.data)/self.P
        self.Sigma_c = self.Sigma_c - np.diag(self.Sigma_c)


    def boltzmann_gibbs(self, x):
        w, theta = self.w, self.theta

        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)

        unnormalized = np.exp(.5*np.einsum('ij,ik,jk->k', w, x, x) + np.einsum('i,ik->k', theta, x))

        return unnormalized


    def boltzmann_gibbs_normalized(self, x):
        unnormalized = self.boltzmann_gibbs(x)
        Z = self.partition_function()

        out = unnormalized/Z
        if len(out) == 1:
            out = out[0]
        return out
    

    @cached_property
    def all_configurations(self):
        all_configs = list(product([-1, 1], repeat=self.N))
        return np.array(all_configs).T
    
    
    def partition_function(self):
        if self.N <= 20 and self.approx_partition == False:
            Z = np.sum(self.boltzmann_gibbs(self.all_configurations))
        else:
            ## TODO: Not sure how to but here we need to replace self.mu_c by some different m_i
            ## THIS DOES NOT WORK YET!
            # For now:
            m_i = self.mu_c
            # See handout p. 19/20
            log_term = np.sum((1+m_i) * np.log(.5*(1+m_i)) + (1-m_i) *  np.log(.5*(1-m_i)))
            F = -.5 * np.einsum("ij,i,j->", self.w, m_i, m_i) + .5 * log_term
            Z = np.exp(-F)
            print(Z)
        return Z


    def log_likelihood(self, x):
        prob = self.boltzmann_gibbs_normalized(x)
        if np.isscalar(prob):
            prob = np.array([prob])
        return np.einsum("i->",np.log(prob))/self.P
    

    def train(self, method, train_weights=True):

        change = 1

        # for _ in range(500):
        while True: 

            if method == "MH":
                self.mu, Sigma = self.metropolis_hastings(self.num_iter)
            if method == "exact":
                all_probs = self.boltzmann_gibbs_normalized(self.all_configurations)
                # assert(np.isclose(np.sum(all_probs),1))
                
                self.mu = self.all_configurations @ all_probs
                Sigma = np.einsum('ik,k,jk->ij',self.all_configurations, all_probs, self.all_configurations) 
            if method == "MF":
                self.mu, Sigma = self.mf()
            Sigma = Sigma - np.diag(Sigma)
            # assert np.all(np.isclose(Sigma, Sigma.T))

            dw = (self.Sigma_c - Sigma)
            dtheta = (self.mu_c - self.mu)

            if train_weights:
                self.w += self.eta * dw
            else:
                self.w = np.zeros_like(self.w)
            self.theta += self.eta * dtheta
            # assert np.all(np.isclose(self.w, self.w.T))

            self.all_LLs[method].append(self.log_likelihood(self.data))

            if method == "exact":
                change = self.eta * np.max((np.max(np.abs(dw)), np.max(np.abs(dtheta))))
                if change < self.epsilon:
                    break
                print(f'max param change:{change}')

            if method == "MH":
                print(np.mean(self.all_LLs["MH"]))
                if len(self.all_LLs["MH"]) < 100:
                    continue
                # print(np.mean(self.all_LLs["MH"]))
                print(np.abs(np.mean(self.all_LLs["MH"][-300:]) - np.mean(self.all_LLs["MH"][-50:])))
                if np.abs(np.mean(self.all_LLs["MH"][-300:]) - np.mean(self.all_LLs["MH"][-50:])) < 1e-1:
                    break

            if method == "MF":
                print(np.mean(self.all_LLs["MF"]))
                change = self.eta * np.max((np.max(np.abs(dw)), np.max(np.abs(dtheta))))
                if change < self.epsilon:
                    break
                if len(self.all_LLs["MF"]) < 100:
                    continue
                # print(np.mean(self.all_LLs["MF"]))
                print(np.abs(np.mean(self.all_LLs["MF"][-300:]) - np.mean(self.all_LLs["MF"][-50:])))
                if np.abs(np.mean(self.all_LLs["MF"][-300:]) - np.mean(self.all_LLs["MF"][-50:])) < 1e-1:
                    break

        print("Converged.")


    def spin_flip(self, state, flip_loc):
        x = state.copy()
        x[flip_loc] *= -1
        return x
    

    # calc energy difference of spin flip for BM
    def energy_diff(self, x, flip_loc):
        return 2 * x[flip_loc] * (self.w[flip_loc,:]@x + self.theta[flip_loc])


    def metropolis_hastings(self, num_samples):
        initial_state = np.random.randn(self.N)
        samples = [initial_state]

        acceptance_ratio=np.empty(num_samples)
        accepted = []

        for i in range(num_samples-1):
            current_state = samples[-1]

            # Sample random spin flip location
            flip_loc = np.random.randint(0, self.N - 1)

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
        self.mu = np.mean(samples, axis = 1)
        Sigma = np.einsum("ik, jk -> ij", samples, samples)/num_samples

        return self.mu, Sigma #, current_state
    
    def mf(self, meth="seq"):
        couplings = self.w
        field = self.theta
        mf_convergence_threshold = self.epsilon
        n_spins = self.N
        spin_means = np.random.uniform(-1, 1, n_spins)
        indexes = np.array(range(spin_means.size))
        max_diff_spin = 1e10
        while max_diff_spin > mf_convergence_threshold:
            old_spin_means = np.copy(spin_means)
            if meth == "par":
                spin_means = np.tanh(np.einsum('ji,i->j', couplings, spin_means)+field)
            if meth == "seq":
                for i in indexes:
                    null_inxs = np.concatenate([indexes[:i], indexes[i+1:] ])
                    interaction = np.dot(spin_means[null_inxs], couplings[i, null_inxs])
                    spin_means[i] = np.tanh(interaction + field[i])
            max_diff_spin = np.max(np.abs(old_spin_means - spin_means))
        self.mu = spin_means
        fraction = np.diag(1/ (1 - np.einsum('i, i-> i', self.mu, self.mu)))
        A_inverse = fraction - couplings
        sigma = np.linalg.inv(A_inverse) + np.einsum('i,j->ij', spin_means, spin_means)
        return self.mu, sigma

    def plot_LL(self, out, method = "exact", train_weights=True):

        if len(self.all_LLs[method]) == 0:
            self.train(method, train_weights)
        #TODO: THIS OF COURSE NEEDS TO BE DONE PROPERLY HAHA

        plt.figure()
        plt.plot(range(len(self.all_LLs[method])), self.all_LLs[method])
        plt.xlabel('Runs')
        plt.ylabel('Log Likelihood')
        plt.grid()
        plt.savefig(f"plots/LL_{out}_{method}.png")

    
    def solve_fixed_points(self, epsilon_1):
        
        self.mu = self.mu_c.copy()
        C = self.Sigma_c - np.outer(self.mu_c, self.mu_c)

        C_inv = np.linalg.inv(C)
        C_inv = C_inv + epsilon_1*np.eye(self.N)

        self.w = np.eye(self.N)/(1 - self.mu_c) - C_inv
        self.w = np.divide(np.eye(self.N),(1 - self.mu_c)[:, np.newaxis]) - C_inv
        self.theta = np.arctanh(self.mu_c) - self.w @ self.mu_c

        print("Log Likelihood after: ", self.log_likelihood(self.data))


def main():
    # set params
    n = 50 # number of spins
    p = 50 # number of patterns
    eta = .005 # learning rate
    epsilon = 1e-13 # convergence criterion
    num_iter = 1000 # number of MH steps

    # generate random data set with 10-20 spins
    x_small = np.random.choice([-1, 1], size=(n,p))
    w_init = np.ones((n,n))
    w = w_init = w_init - np.diag(w_init)
    theta = theta_init = np.random.randn(n)

    bm = BoltzmannMachine(x_small, w, theta, eta, epsilon, num_iter)
    bm.plot_LL("toydata", "MH")

    for epsilon_1 in 10.**(np.arange(1, -2, -.2)):
        print(epsilon_1)
        bm.solve_fixed_points(epsilon_1)
        print("test")

if __name__ == "__main__":
    main()
