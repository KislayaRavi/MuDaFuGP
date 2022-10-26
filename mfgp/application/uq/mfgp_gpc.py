import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class MFGP_GPC(object):

    def __init__(self, mfgp_obj: callable, gpc_obj: callable, num_adapts: int, init_cost:float, X_test: np.ndarray=None, Y_test: np.ndarray = None):
        """
        :param mfgp_obj: multifidelity gaussian process object cretaed with data fusion between low and high fidelity data
        :type mfgp_obj: GPDF object (callable)

        :param gpc_obj: Polynomial Chaos expansion of the mfgp_obj, created using chaospy
        :type gpc_obj: ChaospyWrapper object (callable)
        """

        self.mfgp_obj, self.num_adapts, self.gpc_obj = mfgp_obj, num_adapts, gpc_obj
        self.gpc_obj.calculate_coefficients()
        self.mean_history, self.var_history = [self.gpc_obj.get_mean()], [self.gpc_obj.get_var()]

        self.adapt_per_steps = 5
        # self.adapt_per_steps = np.array([6 , 4, 2])

        self.cost_history = [init_cost]
        self.X_test, self.Y_test, self.calculate_mse = X_test, Y_test, False
        if (self.X_test is not None) and (self.Y_test is not None):
            self.calculate_mse = True
            self.mse_history = [self.mfgp_obj.get_mse(self.X_test, self.Y_test)]

    def adapt(self):

        for i in range(self.num_adapts):

            print("Step", i+1)
            #find the function value with largest variane and add it to the training set to improve MFGP accurance
            self.mfgp_obj.adapt(self.adapt_per_steps) # <<<<<<<<<<<<<<<< the function call that includes the optimization step <<<<<<<<<<<<<<<<
            
            # Use the predict function of the MFGP as an input function for the chaospy polynomial chaos expansion methodology
            temp_f = lambda x: self.mfgp_obj.predict(x)[0] 

            # Calculate the polynomial chaos expansion of the MFGP after adaptation using Chaospy
            self.gpc_obj.update_function(temp_f)

            self.mean_history.append(self.gpc_obj.get_mean())
            self.var_history.append(self.gpc_obj.get_var())
            if type(self.adapt_per_steps) is int:
                self.cost_history.append(self.cost_history[-1] + self.mfgp_obj.adapt_steps)
            else:
                self.cost_history.append(self.cost_history[-1] + self.mfgp_obj.adapt_steps.sum())

            if self.calculate_mse:
                self.mse_history.append(self.mfgp_obj.get_mse(self.X_test, self.Y_test))
