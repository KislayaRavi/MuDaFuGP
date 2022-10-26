import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import product

'''Source : https://www.sfu.ca/~ssurjano/optimization.html
'''



class AbstractBenchmark(metaclass=ABCMeta):

    def __init__(self, dim):
        self._dim = dim

    def get_dim(self):
        return self._dim

    def evaluate_hf(self, parameter):
        X = np.atleast_2d(parameter)
        assert X.shape[1] == self._dim, "Incorrect dimension"
        Y = np.zeros((X.shape[0],1))
        for idx, x in enumerate(X):
            Y[idx, 0] = self.evaluate_one_hf(x)
        return Y 

    def evaluate_lf(self, parameter):
        X = np.atleast_2d(parameter)
        assert X.shape[1] == self._dim, "Incorrect dimension"
        Y = np.zeros((X.shape[0], 1))
        for idx, x in enumerate(X):
            Y[idx, 0] = self.evaluate_one_lf(x)
        return Y 

    def plot(self, category='hf'):
        if self._dim == 1:
            self.plot_1d(category=category)
        elif self._dim == 2:
            self.plot_2d(category=category)
        else:
            raise ValueError("Cannot plot for more that 2 dimensions")
        plt.show()

    def plot_1d(self, num_mesh=100, category='hf'):
        lower, upper = self.get_bounds()
        x = np.linspace(lower, upper, num_mesh)
        if category=='hf':
            y = self.evaluate_hf(x)
        elif category=='lf':
            y = self.evaluate_lf(x)
        else:
            y = self.evaluate_hf(x) - self.evaluate_lf(x)
        plt.plot(x, y)
        plt.title(self.get_name())
        plt.xlabel(r'$x$')
        plt.ylabel(r'$f(x)$')

    def plot_2d(self, num_mesh=100, category='hf'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        lower, upper = self.get_bounds()
        x = np.linspace(lower[0], upper[0], num_mesh)
        y = np.linspace(lower[1], upper[1], num_mesh)
        X, Y = np.meshgrid(x, y)
        temp = np.array(list(product(x,y)))
        if category=='hf':
            zs = self.evaluate_hf(temp)
        elif category=='lf':
            zs = self.evaluate_lf(temp)
        else:
            zs = self.evaluate_hf(temp) - self.evaluate_lf(temp)
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(r'$f(x_1, x_2)$')
        ax.set_title(self.get_name())
    
    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def evaluate_one_hf(self, x):
        pass

    @abstractmethod
    def evaluate_one_lf(self, x):
        pass 

    @abstractmethod
    def get_global_optimum(self):
        pass 

    @abstractmethod
    def get_bounds(self):
        pass 



class AckleyFunction(AbstractBenchmark):
    '''The function poses a risk for optimization algorithms, particularly hillclimbing algorithms, 
    to be trapped in one of its many local minima.
    Recommended variable values are: a = 20, b = 0.2 and c = 2*np.pi
    '''

    def __init__(self, dim, a=20, b=0.2, c=2*np.pi):
        super().__init__(dim)
        self.a, self.b, self.c = a, b, c 
        self.inv_d = 1/dim
        self.constant = self.a + np.exp(1)

    def get_name(self):
        return "Ackley Function"

    def evaluate_one_hf(self, x):
        t1 = -self.a * np.exp(-self.b * np.sqrt(self.inv_d * np.sum(x**2)))
        t2 = -np.exp(self.inv_d * np.sum(np.cos(self.c * x)))
        return t1 + t2 + self.constant 

    def evaluate_one_lf(self, x):
        t1 = -self.a* np.exp(-self.b * np.sqrt(self.inv_d * np.sum(x**2)))
        t2 = -np.exp(self.inv_d * np.sum(np.cos(self.c * x)))
        return 1.1*t1 + 0.9*t2 + self.constant

    def get_global_optimum(self):
        return np.zeros(self._dim)

    def get_bounds(self):
        return np.ones(self._dim)*-10, np.ones(self._dim)*10


class BukinFunction_N6(AbstractBenchmark):
    '''The sixth Bukin function has many local minima, all of which lie in a ridge. 
    '''

    def __init__(self):
        super().__init__(2)

    def get_name(self):
        return "Bukin Function number 6"

    def evaluate_one_hf(self, x):
        return 100 * np.sqrt(np.abs(x[1] - 0.01*x[1]*x[1])) + 0.01*np.abs(x[0] + 10)

    def evaluate_one_lf(self, x):
        return 98 * np.sqrt(np.abs(x[1] - 0.01*x[1]*x[1])) + 0.01*np.abs(x[0] + 9.9)

    def get_global_optimum(self):
        return np.array([-10, 1])

    def get_bounds(self):
        return np.array([-15, -3]), np.array([-3, 3])


class DropWave(AbstractBenchmark):
    '''Mutltimodal and highly complex function
    '''

    def __init__(self):
        super().__init__(2)

    def get_name(self):
        return "Drop-Wave function"

    def evaluate_one(self, x):
        return -(1+ np.cos(12*np.sqrt(x[0]**2 + x[1]**2))) / (2 + 0.5*(x[0]**2 + x[1]**2))

    def get_global_optimum(self):
        return np.array([0, 0])

    def get_bounds(self):
        return np.array([-2, -2]), np.array([2, 2])


class Rastrigin(AbstractBenchmark):
    '''The Rastrigin function has several local minima. 
    It is highly multimodal, but locations of the minima are regularly distributed.
    '''

    def __init__(self, dim):
        super().__init__(dim)

    def get_name(self):
        return "Rastrigin function"

    def evaluate_one(self, x):
        return 10 * self._dim + np.sum(x**2 - 10*np.cos(2*np.pi*x))

    def get_global_optimum(self):
        return np.array([0, 0])

    def get_bounds(self):
        return np.array([-3, -3]), np.array([3, 3])


class Bohachevsky1(AbstractBenchmark):
    '''Bohachevsky function is bowl shaped function
    '''

    def __init__(self):
        super().__init__(2)

    def get_name(self):
        return "Bohachevsky function 1"

    def evaluate_one(self, x):
        return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) + 0.7

    def get_global_optimum(self):
        return np.array([0, 0])

    def get_bounds(self):
        return np.array([-3, -3]), np.array([3, 3])


class Bohachevsky2(AbstractBenchmark):
    '''Bohachevsky function is bowl shaped function
    '''

    def __init__(self):
        super().__init__(2)

    def get_name(self):
        return "Bohachevsky function 2"

    def evaluate_one(self, x):
        return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0])*np.cos(4*np.pi*x[1]) + 0.3

    def get_global_optimum(self):
        return np.array([0, 0])

    def get_bounds(self):
        return np.array([-3, -3]), np.array([3, 3])


class Bohachevsky3(AbstractBenchmark):
    '''Bohachevsky function is bowl shaped function
    '''

    def __init__(self):
        super().__init__(2)

    def get_name(self):
        return "Bohachevsky function 3"

    def evaluate_one(self, x):
        return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0] + 4*np.pi*x[1]) + 0.3

    def get_global_optimum(self):
        return np.array([0, 0])

    def get_bounds(self):
        return np.array([-3, -3]), np.array([3, 3])


class Zakharov(AbstractBenchmark):
    '''Zakharov function is plate shaped function
    It has only one optimum at (0,0, ..d times)
    '''

    def __init__(self, dim):
        super().__init__(dim)

    def get_name(self):
        return "Zakharov function"

    def evaluate_one(self, x):
        temp = np.sum([0.5*(i+1)*x[i] for i in range(self._dim)])
        return np.sum(x**2) + temp**2 + temp**4 

    def get_global_optimum(self):
        return np.array([0, 0])

    def get_bounds(self):
        return np.array([-3, -3]), np.array([3, 3])


class SixHumpCamel(AbstractBenchmark):
    '''Bohachevsky function is bowl shaped function
    '''

    def __init__(self):
        super().__init__(2)

    def get_name(self):
        return "Six Hump Camel function"

    def evaluate_one_hf(self, x):
        return ((4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (4*x[1]**2 - 4)*x[1]**2)*-1.0

    def evaluate_one_lf(self, x):
        return ((4 - 2.2*x[0]**2 + x[0]**4/3.2)*x[0]**2 + 1.1*x[0]*x[1] + (4.1*x[1]**2 - 4)*x[1]**2)*1.0

    def get_global_optimum(self):
        return np.array([0.0898, -0.7126]), np.array([-0.0898, 0.7126])

    def get_bounds(self):
        return np.array([-1, -1]), np.array([1, 1])


class Easom(AbstractBenchmark):
    '''Easom function has several local minimum.
    The global minimum has a small area in relative search space
    '''

    def __init__(self):
        super().__init__(2)

    def get_name(self):
        return "Easom function"

    def evaluate_one(self, x):
        return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 -(x[1] - np.pi)**2)

    def get_global_optimum(self):
        return np.array([np.pi, np.pi])

    def get_bounds(self):
        return np.array([0, 0]), np.array([6, 6])


class Branin(AbstractBenchmark):
    '''The Branin, or Branin-Hoo, function has three global minima. 
    The recommended values of a, b, c, r, s and t are: 
    a = 1, b = 5.1 ⁄ (4*np.pi*np.pi), c = 5 ⁄ np.pi, r = 6, s = 10 and t = 1 ⁄ (8*np.pi)

    TODO: Look into Picheny et al (2012) and implement that version
    '''

    def __init__(self, dim, a=20, b=0.2, c=2*np.pi):
        super().__init__(dim)
        self.a, self.b, self.c = a, b, c 
        self.inv_d = 1/dim
        self.constant = self.a + np.exp(1)

    def get_name(self):
        return "Ackley Function"

    def evaluate_one(self, x):
        t1 = -self.a * np.exp(-self.b * np.sqrt(self.inv_d * np.sum(x**2)))
        t2 = -np.exp(self.inv_d * np.sum(np.cos(self.c * x)))
        return t1 + t2 + self.constant 

    def get_global_optimum(self):
        return np.zeros(self._dim)

    def get_bounds(self):
        return np.ones(self._dim)*-10, np.ones(self._dim)*10


if __name__ == '__main__':
    # benchmark = AckleyFunction(2, a=20, b=0.2, c=2*np.pi)
    # benchmark = BukinFunction_N6()
    # benchmark = DropWave()
    # benchmark = Rastrigin(2)
    # benchmark = Bohachevsky1()
    # benchmark = Bohachevsky2()
    # benchmark = Zakharov(2)
    benchmark = SixHumpCamel()
    # benchmark = Easom()
    benchmark.plot(category='lf')
