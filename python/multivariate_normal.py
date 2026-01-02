import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as nm
from svgd import SVGD

class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
    
    def dlnprob(self, theta):
        return -1*np.matmul(theta-nm.repmat(self.mu, theta.shape[0], 1), self.A)
    
class Cauchy:
    def __init__(self, mu):
        self.mu = mu
    def dlnprob(self, theta):
        diff = theta - self.mu
        return -2 * diff / (1 + diff**2)

    
if __name__ == '__main__':
    A = np.array([[0.2260,0.1652],[0.1652,0.6779]])
    mu = np.array([-0.6871,0.8010])
    
    mu_cauchy = 0

    Gaussian_model = MVN(mu, A)
    Cauchy_model = Cauchy(mu_cauchy)
    
    x0 = np.random.normal(0,1, [10,2])
    theta_gauss = SVGD().update(x0, Gaussian_model.dlnprob, n_iter=1000, stepsize=0.01)
    x0 = np.random.normal(0,1, [1000,1])
    theta_cauchy = SVGD().update(x0, Cauchy_model.dlnprob, n_iter=1000, stepsize=0.01)
    
    print("Gaussian ground truth: ", mu)
    print(theta_gauss.shape)
    print("svgd: ", np.mean(theta_gauss, axis=0))

    print("Cauchy ground truth: ", mu_cauchy)
    print(theta_cauchy.shape)
    print("svgd: ", np.mean(theta_cauchy))

    plt.hist2d(theta_gauss[:,0], theta_gauss[:,1])
    plt.title('SVGD Gaussian Samples')
    plt.show()
    
    plt.hist(theta_cauchy)
    plt.title('SVGD Cauchy Samples')
    plt.show()
