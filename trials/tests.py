import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp 

def lorenz_test():
    """Test Case : Lorenz Attractor. 

    :return: Dataset D:=(xi,yi)

    Simulates the chaotic spiral of the Lorenz Attractor.
    Solved for the initial condtion [x,y,z] = [1,0,0].
    Paramters include [alpha,rho,beta] = [10,28,8/3]
    Forward time stepping is done for 100s. 


    """ 
    def lorenz(t,S,alpha,rho,beta):
        x,y,z  = S
        return np.array([
            alpha*(y-x), 
            x*(rho-z)-y,
            x*y - beta*z
        ])


    u0 = [1,0,0]
    tspan = [0,100] 
    params = (10,28,8/3) 

    solution = solve_ivp(lorenz,tspan,u0,args=params,dense_output=True)

    xdata = solution.t.reshape(-1,1)
    ydata = solution.y.T

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    plt.plot(ydata[:,0],ydata[:,1],ydata[:,2],lw=0.5)
    plt.title("Lorenz Attractor - Phase space")
    return (xdata,ydata)


def vanilla_test():
    N = 100 
    X = np.random.rand(N)[:, None] * 10 - 5  # Inputs = N x D
    G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))  # G = N x L
    W = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])  # L x P
    F = np.matmul(G, W)  # N x P
    Y = F + np.random.randn(*F.shape) * [0.2, 0.2, 0.2]

    return X, Y