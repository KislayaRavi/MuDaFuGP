import gpflow as gpf
import numpy as np

#---------------------------------------------------------------------------------------------------------------#
class MOGP():
    """Multi-Output Gaussian Processes. 

    Is an abstract class for MOGP methods that follow. 
    The methods __init__, predict, visualize are to be defined for every other method that is included. 
    """
    def __init__(self,data): 
        """Common Constructor for all MOGP methods.
        """
        self.x = data[0] 
        self.y = data[1]
        assert type(self.x) == np.ndarray and type(self.y) == np.ndarray , "Work with numpy ndarrays for efficiency." 
        assert self.x.shape[0] == self.y.shape[0] , "In D ~ (x_i,y_i), i != 1:N" 
        self.nsamples = self.x.shape[0]
        self.nxdims = self.x.ndim
        self.nydims = self.y.ndim
        self.nparams = self.y.shape[1]
 
    def kernel(self):
        """Abstract Kernel Method.
        """ 
        pass 

    def predict(self): 
        """Abstract Prediction Method.
        """
        pass 

    def visualize(self): 
        """Abstract Visualization Method.
        """
        pass
#---------------------------------------------------------------------------------------------------------------# 

class IndependentGPs(MOGP): 
    """Independent Gaussian Processes. 

    Every dimension of the data, makes use of an independent Gaussian process for inference. 
    This assumes that the columns of the data are not correlated in any way. 
    This is less useful, however an implementation exists here for completeness.
    """
    def __init__(self,data,kernels=[]): 
        MOGP.__init__(self,data)
        self.models = None
        self.means = []
        self.vars = [] 
        self.kernel(kernels)

    def kernel(self,kernels=[]): 
        if len(kernels) == 0:
            self.kernels = [gpf.kernels.RBF() for i in range(self.nparams)]
        else:
            self.kernels = kernels

    def predict(self,Xtest):
        m = lambda i : gpf.models.GPR(data=(self.x,self.y[:,i]),kernel=self.kernels[i],mean_function=None)  
        self.models = [m(i) for i in range(self.nparams)]
        opt = gpf.optimizers.Scipy()
        for i in range(self.nparams):
            opt.minimize(self.models[i].training_loss,self.models[i].trainable_variables,options=dict(maxiter=100))

        for i in range(self.nparams):
            mean,var = self.models[i].predict_f(Xtest.reshape(-1,1))
            self.means.append(mean)
            self.vars.append(var)

        return self.means,self.vars

    def __str__(self): 
        pass 

    def visualize(self): 
        pass 
#---------------------------------------------------------------------------------------------------------------#

class ICM(MOGP):
    """Intrinsic Coregionalization Model
    """ 
    def __init__(self,data,n_induced,kernels=[]):
        MOGP.__init__(self,data)
        Z = np.linspace(self.x.min(), self.x.max(),int(n_induced)).reshape(-1,1)
        self.IV = gpf.inducing_variables.SharedIndependentInducingVariables(gpf.inducing_variables.InducingPoints(Z))
        self.means = None
        self.vars = None
        self.kernel(kernels)

    def kernel(self,kernels):
        if len(kernels) == 0:
            self.kernels = gpf.kernels.SharedIndependent(gpf.kernels.RBF()+gpf.kernels.Linear(),output_dim=self.nparams) 
        else: 
            self.kernels = gpf.kernels.SharedIndependent(kernels[0],output_dim=self.nparams)

    def predict(self,Xtest): 
        self.model = gpf.models.SVGP(self.kernels,gpf.likelihoods.Gaussian(),inducing_variable=self.IV, num_latent_gps=self.nparams)
        opt = gpf.optimizers.Scipy()
        opt.minimize(self.model.training_loss_closure((self.x,self.y)),
        variables=self.model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": 1000},)
        self.mean,self.vars = self.model.predict_y(Xtest.reshape(-1,1))
        return self.mean,self.vars

    def __str__(self): 
        pass

    def visualize(self): 
        pass 
#---------------------------------------------------------------------------------------------------------------#

class SLFM(MOGP): 
    """Semi-parametric Latent Factor Model
    """
    def __init__(self,data,n_induced,kernels=[]):
        MOGP.__init__(self,data)
        Z = np.linspace(self.x.min(), self.x.max(),int(n_induced)).reshape(-1,1)
        self.IV = gpf.inducing_variables.SharedIndependentInducingVariables(gpf.inducing_variables.InducingPoints(Z))
        self.means = None
        self.vars = None 
        self.kernel(kernels)

    def kernel(self,kernels):
        if len(kernels) == 0:
            self.kernels =  gpf.kernels.SeparateIndependent([gpf.kernels.SquaredExponential() for _ in range(self.nparams)])
        else: 
            self.kernels = gpf.kernels.SeparateIndependent(kernels)

    def predict(self,Xtest):
        self.model =  gpf.models.SVGP(self.kernels,gpf.likelihoods.Gaussian(),inducing_variable=self.IV,num_latent_gps=self.nparams)
        opt = gpf.optimizers.Scipy()
        opt.minimize(self.model.training_loss_closure((self.x,self.y)),
        variables=self.model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": 1000},)
        self.mean,self.vars = self.model.predict_y(Xtest.reshape(-1,1))
        return self.mean,self.vars

    def __str__(self):
        pass 

    def visualize(self):
        pass

#---------------------------------------------------------------------------------------------------------------#

class LCM(MOGP): 
    """Linear Congruential Model 
    """
    def __init__(self,data,n_induced,nGPs,kernels=[]):
        MOGP.__init__(self,data)
        self.nGPs = nGPs
        self.nInduced = n_induced
        Z = np.linspace(self.x.min(), self.x.max(),int(n_induced)).reshape(-1,1)
        self.IV = gpf.inducing_variables.SharedIndependentInducingVariables(gpf.inducing_variables.InducingPoints(Z))
        self.means = None
        self.vars = None 
        self.kernel(kernels)

    def kernel(self,kernels): 
        if len(kernels) == 0: 
            kernel_list = [gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(self.nGPs)]
            self.kernels = gpf.kernels.LinearCoregionalization(kernel_list,W=np.random.randn(self.nparams,self.nGPs))
        else: 
            self.kernels = gpf.kernels.LinearCoregionalization(kernels,W=np.random.randn(self.nparams,self.nGPs))

    def predict(self,Xtest):
        q_mu = np.zeros((self.nInduced , self.nGPs))
        q_sqrt = np.repeat(np.eye(self.nInduced)[None, ...], self.nGPs, axis=0) * 1.0
        self.model = gpf.models.SVGP(
            self.kernels, gpf.likelihoods.Gaussian(), inducing_variable=self.IV, q_mu=q_mu, q_sqrt=q_sqrt
            ) 
        opt = gpf.optimizers.Scipy()
        opt.minimize(self.model.training_loss_closure((self.x,self.y)),
        variables=self.model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": 1000},)
        self.mean,self.vars = self.model.predict_y(Xtest.reshape(-1,1))
        return self.mean,self.vars
        

    def __str__(self):
        pass 

    def visualize(self):
        pass
#---------------------------------------------------------------------------------------------------------------#