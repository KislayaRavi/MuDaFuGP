import gpflow
import tensorflow as tf

class SquaredExponential(gpflow.kernels.IsotropicStationary):
    """
    The radial basis function (RBF) or squared exponential kernel. The kernel equation is
        k(r) = σ² exp{-½ r²}
    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ²  is the variance parameter
    Functions drawn from a GP with this kernel are infinitely differentiable!
    """

    def K_r2(self, r2):
        r = tf.maximum(r2, 1e-16)
        return self.variance * tf.exp(-0.5 * r)

    def scale(self, X):
        X_scaled = X / (self.lengthscales + 1e-16) if X is not None else X
        return X_scaled