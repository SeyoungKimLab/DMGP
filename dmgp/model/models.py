import sys
sys.path.append('../../') # Add DMGP root

import copy
from functools import partial, update_wrapper
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

import gpflow
from gpflow import kullback_leiblers
from gpflow.kullback_leiblers import gauss_kl
from gpflow.models import GPModel
from gpflow.models.util import inducingpoint_wrapper
from gpflow.kernels import SquaredExponential
from gpflow.likelihoods import Gaussian
from gpflow.inducing_variables import InducingPoints
from gpflow.base import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.utilities import triangular, positive, to_default_float

import dmgp
from dmgp.core import vec, vec_inv, KP

gpflow.config.set_default_jitter(1e-6)

# Doubly Mixed-Effects Gaussian Process (DMGP)
class DMGP(GPModel):
    def __init__(self, shape, ids, kernel, ll, ind_points, initial_cov='prior'):
        super().__init__(kernel=kernel, likelihood=ll, num_latent_gps=1)

        self.p, self.n = shape
        self.G_ids, self.X_ids = ids
        
        # Inducing points
        self.ZX_fixed = inducingpoint_wrapper(ind_points[0])
        self.ZX_random = inducingpoint_wrapper(ind_points[1])
        self.ZG_fixed = inducingpoint_wrapper(ind_points[2])
        self.ZG_random = inducingpoint_wrapper(ind_points[3])
        
        self.mX_fixed = ind_points[0].num_inducing
        self.mX_random = ind_points[1].num_inducing
        self.mG_fixed = ind_points[2].num_inducing
        self.mG_random = ind_points[3].num_inducing

        self.n_ind_points = {'ZX_fixed': self.mX_fixed,
                             'ZX_random': self.mX_random,
                             'ZG_fixed': self.mG_fixed,
                             'ZG_random': self.mG_random}
        
        # Kernel functions
        self.KX_fixed = copy.deepcopy(kernel) # Prior fixed effect kernel (X)
        self.KX_random = copy.deepcopy(kernel) # Prior random effect kernel (X)
        self.KG_fixed = copy.deepcopy(kernel) # Prior fixed effect kernel (G)
        self.KG_random = copy.deepcopy(kernel) # Prior random effect kernel (G)
        
        # Disable training for unused parameters
        gpflow.set_trainable(self.kernel.parameters[0], False)
        gpflow.set_trainable(self.kernel.parameters[1], False)

        # Initialize variational parameters
        if initial_cov == 'prior':
            self.L_uX_fixed = Parameter(self.prior_cov(self.KX_fixed, self.ZX_fixed), transform=triangular()) # [m,m]
            self.L_uX_random = Parameter(tf.repeat(self.prior_cov(self.KX_random, self.ZX_random)[tf.newaxis,...], self.p, axis=0),
                                         transform=triangular()) # [p,mX,mX]
            self.L_uG_fixed = Parameter(self.prior_cov(self.KG_fixed, self.ZG_fixed), transform=triangular()) # [m,m]
            self.L_uG_random = Parameter(tf.repeat(self.prior_cov(self.KG_random, self.ZG_random)[tf.newaxis,...], self.n, axis=0),
                                         transform=triangular()) # [n,mG,mG]

        elif initial_cov == 'identity':
            self.L_uX_fixed = Parameter(np.eye(self.mX_fixed, dtype=default_float()), transform=triangular())
            self.L_uX_random = Parameter(tf.repeat(np.eye(self.mX_random, dtype=default_float())[tf.newaxis,...], self.p, axis=0),
                                         transform=triangular()) # [p,mX,mX]
            self.L_uG_fixed = Parameter(np.eye(self.mG_fixed, dtype=default_float()), transform=triangular())
            self.L_uG_random = Parameter(tf.repeat(np.eye(self.mG_random, dtype=default_float())[tf.newaxis,...], self.n, axis=0),
                                         transform=triangular()) # [n,mG,mG]

        else:
            raise RuntimeError(f'[Error] Invalid covariance initialization: {initial_cov}')

        self.m_uX_fixed = Parameter(np.zeros((self.mX_fixed,1), dtype=default_float())) # [mX,1]
        self.m_uX_random = Parameter(np.zeros((self.p,self.mX_random,1), dtype=default_float())) # [p,mX,1]
        self.m_uG_fixed = Parameter(np.zeros((self.mG_fixed,1), dtype=default_float())) # [mG,1]
        self.m_uG_random = Parameter(np.zeros((self.n,self.mG_random,1), dtype=default_float())) # [n,mG,1]

    # Computes the prior covariance for initialization
    def prior_cov(self, K, Z, jitter=default_jitter()):
        cov = K(Z.Z, Z.Z)
        cov += jitter * tf.eye(Z.Z.shape[0], dtype=default_float())

        return tf.linalg.cholesky(cov)

    # KL divergence term for fixed effect inducing points
    def fixed_prior_kl(self, jitter=default_jitter()):
        KX = self.KX_fixed(self.ZX_fixed.Z, self.ZX_fixed.Z)\
             + jitter * tf.eye(self.mX_fixed, dtype=default_float())
        KG = self.KG_fixed(self.ZG_fixed.Z, self.ZG_fixed.Z)\
             + jitter * tf.eye(self.mG_fixed, dtype=default_float())

        kl = kullback_leiblers.gauss_kl(
            self.m_uX_fixed, self.L_uX_fixed[tf.newaxis,:], K=KX
        )
        kl += kullback_leiblers.gauss_kl(
            self.m_uG_fixed, self.L_uG_fixed[tf.newaxis,:], K=KG
        )

        return kl

    # Total KL divergence
    def total_prior_kl(self, jitter=default_jitter()):
        # Fixed KL
        kl = self.fixed_prior_kl()

        # Random KL
        KX = self.KX_random(self.ZX_random.Z, self.ZX_random.Z)\
             + jitter * tf.eye(self.mX_random, dtype=default_float())
        KG = self.KG_random(self.ZG_random.Z, self.ZG_random.Z)\
             + jitter * tf.eye(self.mG_random, dtype=default_float())

        kl += kullback_leiblers.gauss_kl(tf.transpose(tf.squeeze(self.m_uX_random, axis=-1)),
                                         self.L_uX_random, K=KX)
        kl += kullback_leiblers.gauss_kl(tf.transpose(tf.squeeze(self.m_uG_random, axis=-1)),
                                         self.L_uG_random, K=KG)

        return kl

    # Loss function
    def training_loss(self, data, ids, mask=None):
        return -self.maximum_log_likelihood_objective(data, ids, mask=mask)

    # ELBO objective
    def maximum_log_likelihood_objective(self, data, ids, mask=None):
        return self.elbo(data, ids, mask=mask)

    # Computes the ELBO
    # Note: Expects mini-batch sampling to be handled externally
    def elbo(self, data, ids, mask=None):
        # Unpack data-tuple
        X, G, Y = data
        b, a = Y.shape
        G_ids, X_ids = ids
        
        # Compute the parameters of q(f)
        f_mean, f_var = self.predict_f(data, ids, mask=mask)
        
        # Expected log-likelihood
        y = tf.squeeze(vec(Y), axis=-1)
        if mask is not None:
            y = tf.boolean_mask(y, tf.squeeze(vec(mask), axis=-1))
        
        var_exp = self.likelihood.variational_expectations(tf.squeeze(f_mean), 
                                                           tf.squeeze(f_var), 
                                                           y)
        
        # Compute KL term
        kl = self.total_prior_kl()
        
        # Scale
        scale = (self.n * self.p) / (a * b)
        
        return tf.reduce_sum(var_exp) * scale - kl

    # Returns the mean and marginal variance of f for mini-batch/new data
    def predict_f(self, data, ids, mask=None, sep=None, full_cov=False, full_output_cov=False, jitter=default_jitter()):
        # Unpack data-tuple
        X, G, Y = data
        b, a = Y.shape

        G_ids, X_ids = ids

        if mask is not None:
            assert(mask.shape == Y.shape)
            mask = tf.squeeze(vec(mask), axis=-1)

        # Commutation matrix for A KP B where A is [m,m] and B is [n,n]
        # Computes K(n,m), for K(n,m)(A KP B)K(m,n) = B KP A
        @tf.function
        def commutation_mat(m,n):
            K1_idx = tf.reshape(tf.transpose(tf.stack(tf.split(tf.range(m*n), n))), (-1,))

            return K1_idx

        # Helper functions for computing random effects mean and covariance
        def compute_mean_X_k(m_k):
            return tf.matmul(K_ZX_inv_K_ZX_X_random, m_k, transpose_a=True)

        def compute_mean_G_i(m_i):
            return tf.matmul(K_ZG_inv_K_ZG_G_random, m_i, transpose_a=True)

        def compute_cov_X_k(S_k):
            return tf.matmul(tf.matmul(K_ZX_inv_K_ZX_X_random, S_k, transpose_a=True), 
                             K_ZX_inv_K_ZX_X_random)

        def compute_cov_G_i(S_i):
            return tf.matmul(tf.matmul(K_ZG_inv_K_ZG_G_random, S_i, transpose_a=True),
                             K_ZG_inv_K_ZG_G_random)

        # Fixed Effect Mean (X)
        K_ZX_X_fixed = self.KX_fixed(self.ZX_fixed.Z, X)
        K_ZX_fixed = self.KX_fixed(self.ZX_fixed.Z, self.ZX_fixed.Z)\
                     + jitter * tf.eye(self.mX_fixed, dtype=default_float())
        L_ZX_fixed = tf.linalg.cholesky(K_ZX_fixed)
        factor_ZX_X_fixed = tf.linalg.triangular_solve(L_ZX_fixed, K_ZX_X_fixed, lower=True)
        K_ZX_inv_K_ZX_X_fixed = tf.linalg.triangular_solve(tf.transpose(L_ZX_fixed),
                                                           factor_ZX_X_fixed,
                                                           lower=False)
        fX_mean_fixed = tf.matmul(K_ZX_inv_K_ZX_X_fixed, self.m_uX_fixed, transpose_a=True) # [a,1]
        fX_mean_fixed = tf.repeat(fX_mean_fixed[tf.newaxis,:], b, axis=0) # [b,a,1]
        fX_mean_fixed = vec(tf.squeeze(fX_mean_fixed, axis=-1)) # [b,a] --> [a*b,1]

        if mask is not None:
            assert(fX_mean_fixed.shape[0] == mask.shape[0])
            fX_mean_fixed = tf.boolean_mask(fX_mean_fixed, mask) # [?,1]

        # Fixed Effect Mean (G)
        K_ZG_G_fixed = self.KG_fixed(self.ZG_fixed.Z, G)
        K_ZG_fixed = self.KG_fixed(self.ZG_fixed.Z, self.ZG_fixed.Z)\
                     + jitter * tf.eye(self.mG_fixed, dtype=default_float())
        L_ZG_fixed = tf.linalg.cholesky(K_ZG_fixed)
        factor_ZG_G_fixed = tf.linalg.triangular_solve(L_ZG_fixed, K_ZG_G_fixed, lower=True)
        K_ZG_inv_K_ZG_G_fixed = tf.linalg.triangular_solve(tf.transpose(L_ZG_fixed),
                                                           factor_ZG_G_fixed,
                                                           lower=False)
        fG_mean_fixed = tf.matmul(K_ZG_inv_K_ZG_G_fixed, self.m_uG_fixed, transpose_a=True) # [b,1]
        fG_mean_fixed = tf.repeat(fG_mean_fixed, a, axis=1) # [b,a]
        fG_mean_fixed = vec(fG_mean_fixed) # [b,a] --> #[a*b,1]

        if mask is not None:
            assert(fG_mean_fixed.shape[0] == mask.shape[0])
            fG_mean_fixed = tf.boolean_mask(fG_mean_fixed, mask) # [?,1]
        
        # Fixed Effect Covariance (X)
        K_X_fixed = self.KX_fixed(X, X) # [a,a]
        S_X_fixed = tf.matmul(self.L_uX_fixed, self.L_uX_fixed, transpose_b=True) # [mX,mX]

        fX_cov_fixed = K_X_fixed
        fX_cov_fixed += tf.matmul(tf.matmul(K_ZX_inv_K_ZX_X_fixed, S_X_fixed, transpose_a=True),
                                  K_ZX_inv_K_ZX_X_fixed)
        fX_cov_fixed -= tf.matmul(K_ZX_inv_K_ZX_X_fixed, K_ZX_X_fixed, transpose_a=True) # [a,a]

        # Fixed Effect Covariance (G)
        K_G_fixed = self.KG_fixed(G, G) # [b,b]
        S_G_fixed = tf.matmul(self.L_uG_fixed, self.L_uG_fixed, transpose_b=True) # [mG,mG]

        fG_cov_fixed = K_G_fixed
        fG_cov_fixed += tf.matmul(tf.matmul(K_ZG_inv_K_ZG_G_fixed, S_G_fixed, transpose_a=True),
                                  K_ZG_inv_K_ZG_G_fixed)
        fG_cov_fixed -= tf.matmul(K_ZG_inv_K_ZG_G_fixed, K_ZG_G_fixed, transpose_a=True) # [b,b]

        # Random Effect Mean and Covariance (X)
        K_X_random = self.KX_random(X, X) # [a,a]
        K_ZX_X_random = self.KX_random(self.ZX_random.Z, X) # [mX,a]
        K_ZX_random = self.KX_random(self.ZX_random.Z, self.ZX_random.Z)\
                      + jitter * tf.eye(self.mX_random, dtype=default_float()) # [mX,mX]
        L_ZX_random = tf.linalg.cholesky(K_ZX_random)
        factor_ZX_X_random = tf.linalg.triangular_solve(L_ZX_random, K_ZX_X_random, lower=True)
        K_ZX_inv_K_ZX_X_random = tf.linalg.triangular_solve(tf.transpose(L_ZX_random),
                                                            factor_ZX_X_random,
                                                            lower=False)

        OH_bp = dmgp.core.OH(G_ids, self.G_ids) # [b,p], contains zero-vector if there is no match
        mask_X = tf.linalg.diag(tf.reduce_sum(OH_bp, axis=1)) # [b,b]

        m_uX_random = tf.gather(self.m_uX_random, tf.math.argmax(OH_bp, axis=1), axis=0) # [p,mX,1] --> [b,mX,1]
        m_uX_random = tf.matmul(mask_X, tf.squeeze(m_uX_random, axis=-1))[...,tf.newaxis] # [b,mX,1]
        fX_mean_random = tf.map_fn(compute_mean_X_k, m_uX_random) # [b,mX,1] --> [b,a,1]
        fX_mean_random = vec(tf.squeeze(fX_mean_random, axis=-1)) # [b,a,1] --> [a*b,1]

        if mask is not None:
            assert(fX_mean_random.shape[0] == mask.shape[0])
            fX_mean_random = tf.boolean_mask(fX_mean_random, mask) # [?,1]

        L_uX_random = tf.gather(self.L_uX_random, tf.math.argmax(OH_bp, axis=1), axis=0) # [p,mX,mX] --> [b,mX,mX]
        L_uX_random = tf.reshape(tf.matmul(mask_X, tf.reshape(L_uX_random, (b,-1))), # [b,mX**2]
                                 (b,self.mX_random,self.mX_random))
        S_X_random = tf.matmul(L_uX_random, tf.transpose(L_uX_random, perm=[0,2,1])) # [b,mX,mX]
        fX_cov_random = K_X_random # [a,a]
        fX_cov_random += tf.map_fn(compute_cov_X_k, S_X_random) # [b,a,a], broadcast sum
        fX_cov_random -= tf.matmul(K_ZX_inv_K_ZX_X_random, K_ZX_X_random, transpose_a=True) # [b,a,a]

        # Random Effect Mean and Covariance (G)
        K_G_random = self.KG_random(G, G) # [b,b]
        K_ZG_G_random = self.KG_random(self.ZG_random.Z, G) # [mG,b]
        K_ZG_random = self.KG_random(self.ZG_random.Z, self.ZG_random.Z)\
                      + jitter * tf.eye(self.mG_random, dtype=default_float()) # [mG,mG]
        L_ZG_random = tf.linalg.cholesky(K_ZG_random)
        factor_ZG_G_random = tf.linalg.triangular_solve(L_ZG_random, K_ZG_G_random, lower=True)
        K_ZG_inv_K_ZG_G_random = tf.linalg.triangular_solve(tf.transpose(L_ZG_random),
                                                            factor_ZG_G_random,
                                                            lower=False)

        OH_an = dmgp.core.OH(X_ids, self.X_ids) # [a,n]
        mask_G = tf.linalg.diag(tf.reduce_sum(OH_an, axis=1)) # [a,a]

        m_uG_random = tf.gather(self.m_uG_random, tf.math.argmax(OH_an, axis=1), axis=0) # [n,mG,1] --> [a,mG,1]
        m_uG_random = tf.matmul(mask_G, tf.squeeze(m_uG_random, axis=-1))[...,tf.newaxis] # [a,mG,1]
        fG_mean_random = tf.map_fn(compute_mean_G_i, m_uG_random) # [a,mG,1] --> [a,b,1]
        fG_mean_random = vec(tf.transpose(tf.squeeze(fG_mean_random, axis=-1))) # [a,b,1] --> [b,a] --> [a*b,1]

        if mask is not None:
            assert(fG_mean_random.shape[0] == mask.shape[0])
            fG_mean_random = tf.boolean_mask(fG_mean_random, mask)

        L_uG_random = tf.gather(self.L_uG_random, tf.math.argmax(OH_an, axis=1), axis=0) # [n,mG,mG] --> [a,mG,mG]
        L_uG_random = tf.reshape(tf.matmul(mask_G, tf.reshape(L_uG_random, (a,-1))), # [a,mG**2]
                                 (a,self.mG_random,self.mG_random))
        S_G_random = tf.matmul(L_uG_random, tf.transpose(L_uG_random, perm=[0,2,1])) # [a,mG,mG]
        fG_cov_random = K_G_random # [b,b]
        fG_cov_random += tf.map_fn(compute_cov_G_i, S_G_random) # [a,b,b], broadcast sum
        fG_cov_random -= tf.matmul(K_ZG_inv_K_ZG_G_random, K_ZG_G_random, transpose_a=True) # [a,b,b]

        # Mean
        tf.debugging.assert_shapes([
            (fX_mean_fixed, ('n','q')),
            (fX_mean_random, ('n','q')),
            (fG_mean_fixed, ('n','q')),
            (fG_mean_random, ('n','q'))
        ])

        fX_mean = fX_mean_fixed + fX_mean_random
        fG_mean = fG_mean_fixed + fG_mean_random
        f_mean = fX_mean + fG_mean

        if full_cov:
            # Fixed effect covariance
            fX_cov_fixed = KP(fX_cov_fixed, tf.ones((b,b), dtype=default_float())) # [a*b,a*b]
            fG_cov_fixed = KP(tf.ones((a,a), dtype=default_float()), fG_cov_fixed) # [a*b,a*b]

            # Random effect covariance
            fX_cov_random = tf.unstack(fX_cov_random) # List of b [a,a] tensors
            fX_cov_random = [tf.linalg.LinearOperatorFullMatrix(cov_k) for cov_k in fX_cov_random]
            fX_cov_random = tf.linalg.LinearOperatorBlockDiag(fX_cov_random).to_dense() # [b*a,b*a]

            K1_idx = commutation_mat(b,a) # Reordering
            fX_cov_random = tf.gather(fX_cov_random, K1_idx, axis=0)
            fX_cov_random = tf.gather(fX_cov_random, K1_idx, axis=1) # [a*b,a*b]

            fG_cov_random = tf.unstack(fG_cov_random) # List of a [b,b] tensors
            fG_cov_random = [tf.linalg.LinearOperatorFullMatrix(cov_i) for cov_i in fG_cov_random]
            fG_cov_random = tf.linalg.LinearOperatorBlockDiag(fG_cov_random).to_dense() # [a*b,a*b]

            if mask is not None:
                fX_cov_fixed = tf.boolean_mask(fX_cov_fixed, mask)
                fX_cov_fixed = tf.boolean_mask(tf.transpose(fX_cov_fixed), mask)

                fG_cov_fixed = tf.boolean_mask(fG_cov_fixed, mask)
                fG_cov_fixed = tf.boolean_mask(tf.transpose(fG_cov_fixed), mask)

                fX_cov_random = tf.boolean_mask(fX_cov_random, mask)
                fX_cov_random = tf.boolean_mask(tf.transpose(fX_cov_random), mask)

                fG_cov_random = tf.boolean_mask(fG_cov_random, mask)
                fG_cov_random = tf.boolean_mask(tf.transpose(fG_cov_random), mask)

            assert(fX_cov_fixed.shape == fX_cov_random.shape)
            assert(fG_cov_fixed.shape == fG_cov_random.shape)
            assert(fX_cov_fixed.shape == fG_cov_fixed.shape)
            
            fX_cov = fX_cov_fixed + fX_cov_random
            fG_cov = fG_cov_fixed + fG_cov_random
            f_cov = fX_cov + fG_cov

            if sep == 'fixed':
                return ((fX_mean_fixed, fX_mean_random, fG_mean_fixed, fG_mean_random), 
                        (fX_cov_fixed, fX_cov_random, fG_cov_fixed, fG_cov_random))
            else:
                return (f_mean, f_cov)

        else:
            # Fixed effect variance
            fX_var_fixed = tf.linalg.diag_part(fX_cov_fixed)[:,tf.newaxis] # [a,1]
            fX_var_fixed = KP(fX_var_fixed, tf.ones((b,1), dtype=default_float())) # [a*b,1]

            fG_var_fixed = tf.linalg.diag_part(fG_cov_fixed)[:,tf.newaxis] # [b,1]
            fG_var_fixed = KP(tf.ones((a,1), dtype=default_float()), fG_var_fixed) # [a*b,1]

            # Random effect variance
            fX_var_random = tf.linalg.diag_part(fX_cov_random) # [b,a]
            fX_var_random = vec(fX_var_random) # [a*b,1]

            fG_var_random = tf.linalg.diag_part(fG_cov_random) # [a,b]
            fG_var_random = vec(tf.transpose(fG_var_random)) # [a*b,1]

            if mask is not None:
                fX_var_fixed = tf.boolean_mask(fX_var_fixed, mask)
                fG_var_fixed = tf.boolean_mask(fG_var_fixed, mask)
                fX_var_random = tf.boolean_mask(fX_var_random, mask)
                fG_var_random = tf.boolean_mask(fG_var_random, mask)

            tf.debugging.assert_shapes([
                (fX_var_fixed, ('n','q')),
                (fX_var_random, ('n','q')),
                (fG_var_fixed, ('n','q')),
                (fG_var_random, ('n','q'))
            ])

            fX_var = fX_var_fixed + fX_var_random
            fG_var = fG_var_fixed + fG_var_random
            f_var = fX_var + fG_var

            if sep == 'fixed':
                return ((fX_mean_fixed, fX_mean_random, fG_mean_fixed, fG_mean_random), 
                        (fX_var_fixed, fX_var_random, fG_var_fixed, fG_var_random))
            else:
                return (f_mean, f_var)

    # Returns predicted y based on posterior mean and covariance
    def predict_y(self, data, ids, mask=None):
        f_mean, f_var = self.predict_f(data, ids, mask=mask)

        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    # Returns log of the predictive density for given data
    def predict_log_density(self, data, ids, mask=None):
        _, _, Y = data
        f_mean, f_var = self.predict_f(data, ids, mask=mask)

        y = tf.squeeze(vec(Y), axis=-1)
        if mask is not None:
            y = tf.boolean_mask(y, tf.squeeze(vec(mask), axis=-1))
        
        return self.likelihood.predict_log_density(tf.squeeze(f_mean), 
                                                   tf.squeeze(f_var), 
                                                   y)


# Translated Mixed-Effects Gaussian Process (TMGP)
class TMGP(GPModel):
    def __init__(self, shape, ids, kernel, ll, ind_points, initial_cov='prior'):
        super().__init__(kernel=kernel, likelihood=ll, num_latent_gps=1)

        self.p, self.n = shape
        self.G_ids, self.X_ids = ids
        
        # Inducing points
        self.ZX_fixed = inducingpoint_wrapper(ind_points[0])
        self.ZX_random = inducingpoint_wrapper(ind_points[1])
        self.ZG_fixed = inducingpoint_wrapper(ind_points[2])
        
        self.mX_fixed = ind_points[0].num_inducing
        self.mX_random = ind_points[1].num_inducing
        self.mG_fixed = ind_points[2].num_inducing

        self.n_ind_points = {'ZX_fixed': self.mX_fixed,
                             'ZX_random': self.mX_random,
                             'ZG_fixed': self.mG_fixed}
        
        # Kernel functions
        self.KX_fixed = copy.deepcopy(kernel) # Prior fixed effect kernel (X)
        self.KX_random = copy.deepcopy(kernel) # Prior random effect kernel (X)
        self.KG_fixed = copy.deepcopy(kernel) # Prior fixed effect kernel (G)
        
        # Disable training for unused parameters
        gpflow.set_trainable(self.kernel.parameters[0], False)
        gpflow.set_trainable(self.kernel.parameters[1], False)

        # Variational parameters
        if initial_cov == 'prior':
            self.L_uX_fixed = Parameter(self.prior_cov(self.KX_fixed, self.ZX_fixed), transform=triangular()) # [m,m]
            self.L_uX_random = Parameter(tf.repeat(self.prior_cov(self.KX_random, self.ZX_random)[tf.newaxis,...], self.p, axis=0),
                                         transform=triangular()) # [p,mX,mX]
            self.L_uG_fixed = Parameter(self.prior_cov(self.KG_fixed, self.ZG_fixed), transform=triangular()) # [m,m]

        elif initial_cov == 'identity':
            self.L_uX_fixed = Parameter(np.eye(self.mX_fixed, dtype=default_float()), transform=triangular())
            self.L_uX_random = Parameter(tf.repeat(np.eye(self.mX_random, dtype=default_float())[tf.newaxis,...], self.p, axis=0),
                                         transform=triangular()) # [p,mX,mX]
            self.L_uG_fixed = Parameter(np.eye(self.mG_fixed, dtype=default_float()), transform=triangular())

        else:
            raise RuntimeError(f'[Error] Invalid covariance initialization: {initial_cov}')

        self.m_uX_fixed = Parameter(np.zeros((self.mX_fixed,1), dtype=default_float())) # [mX,1]
        self.m_uX_random = Parameter(np.zeros((self.p,self.mX_random,1), dtype=default_float())) # [p,mX,1]
        self.m_uG_fixed = Parameter(np.zeros((self.mG_fixed,1), dtype=default_float())) # [mG,1]

    # Computes the prior covariance for initialization
    def prior_cov(self, K, Z, jitter=default_jitter()):
        cov = K(Z.Z, Z.Z)
        cov += jitter * tf.eye(Z.Z.shape[0], dtype=default_float())

        return tf.linalg.cholesky(cov)

    # KL divergence term for fixed effect inducing points
    def fixed_prior_kl(self, jitter=default_jitter()):
        KX = self.KX_fixed(self.ZX_fixed.Z, self.ZX_fixed.Z)\
             + jitter * tf.eye(self.mX_fixed, dtype=default_float()) # Jitter not added prior to Cholesky in gauss_kl
        KG = self.KG_fixed(self.ZG_fixed.Z, self.ZG_fixed.Z)\
             + jitter * tf.eye(self.mG_fixed, dtype=default_float())

        kl = kullback_leiblers.gauss_kl(
            self.m_uX_fixed, self.L_uX_fixed[tf.newaxis,:], K=KX
        )
        kl += kullback_leiblers.gauss_kl(
            self.m_uG_fixed, self.L_uG_fixed[tf.newaxis,:], K=KG
        )

        return kl

    # Total KL divergence
    def total_prior_kl(self, jitter=default_jitter()):
        # Fixed KL
        kl = self.fixed_prior_kl()

        # Random KL
        KX = self.KX_random(self.ZX_random.Z, self.ZX_random.Z)\
             + jitter * tf.eye(self.mX_random, dtype=default_float())

        kl += kullback_leiblers.gauss_kl(tf.transpose(tf.squeeze(self.m_uX_random, axis=-1)),
                                         self.L_uX_random, K=KX)

        return kl

    # Loss function
    def training_loss(self, data, ids):
        return -self.maximum_log_likelihood_objective(data, ids)

    # ELBO objective
    def maximum_log_likelihood_objective(self, data, ids):
        return self.elbo(data, ids)

    # Computes the ELBO
    # Note: Expects mini-batch sampling to be handled externally
    def elbo(self, data, ids):
        # Unpack data-tuple
        X, G, Y = data
        b, a = Y.shape
        
        # Compute the parameters of q(f)
        f_mean, f_var = self.predict_f(data, ids)
        
        # Expected log-likelihood
        var_exp = self.likelihood.variational_expectations(tf.squeeze(f_mean), 
                                                           tf.squeeze(f_var), 
                                                           tf.squeeze(vec(Y)))
        
        # Compute KL term
        kl = self.total_prior_kl()
        
        # Scale
        scale = (self.n * self.p) / (a * b)
        
        return tf.reduce_sum(var_exp) * scale - kl

    # Returns the mean and marginal variance of 𝑓 for mini-batch/new data
    def predict_f(self, data, ids, sep=None, full_cov=False, full_output_cov=False, jitter=default_jitter()):
        # Unpack data-tuple
        X, G, Y = data
        b, a = Y.shape

        G_ids, X_ids = ids

        # Commutation matrix for A KP B where A is [m,m] and B is [n,n]
        # Computes K(n,m), for K(n,m)(A KP B)K(m,n) = B KP A
        @tf.function
        def commutation_mat(m,n):
            K1_idx = tf.reshape(tf.transpose(tf.stack(tf.split(tf.range(m*n), n))), (-1,))
            
            return K1_idx

        # Helper functions for computing random effects mean and covariance
        def compute_mean_X_k(m_k):
            return tf.matmul(K_ZX_inv_K_ZX_X_random, m_k, transpose_a=True)

        def compute_mean_G_i(m_i):
            return tf.matmul(K_ZG_inv_K_ZG_G_random, m_i, transpose_a=True)

        def compute_cov_X_k(S_k):
            return tf.matmul(tf.matmul(K_ZX_inv_K_ZX_X_random, S_k, transpose_a=True), 
                             K_ZX_inv_K_ZX_X_random)

        def compute_cov_G_i(S_i):
            return tf.matmul(tf.matmul(K_ZG_inv_K_ZG_G_random, S_i, transpose_a=True),
                             K_ZG_inv_K_ZG_G_random)

        # Fixed Effect Mean (X)
        K_ZX_X_fixed = self.KX_fixed(self.ZX_fixed.Z, X)
        K_ZX_fixed = self.KX_fixed(self.ZX_fixed.Z, self.ZX_fixed.Z)\
                     + jitter * tf.eye(self.mX_fixed, dtype=default_float())
        L_ZX_fixed = tf.linalg.cholesky(K_ZX_fixed)
        factor_ZX_X_fixed = tf.linalg.triangular_solve(L_ZX_fixed, K_ZX_X_fixed, lower=True)
        K_ZX_inv_K_ZX_X_fixed = tf.linalg.triangular_solve(tf.transpose(L_ZX_fixed),
                                                           factor_ZX_X_fixed,
                                                           lower=False)
        fX_mean_fixed = tf.matmul(K_ZX_inv_K_ZX_X_fixed, self.m_uX_fixed, transpose_a=True) # [a,1]
        fX_mean_fixed = tf.repeat(fX_mean_fixed[tf.newaxis,:], b, axis=0) # [b,a,1]
        fX_mean_fixed = vec(tf.squeeze(fX_mean_fixed, axis=-1)) # [b,a] --> [a*b,1]

        # Fixed Effect Mean (G)
        K_ZG_G_fixed = self.KG_fixed(self.ZG_fixed.Z, G)
        K_ZG_fixed = self.KG_fixed(self.ZG_fixed.Z, self.ZG_fixed.Z)\
                     + jitter * tf.eye(self.mG_fixed, dtype=default_float())
        L_ZG_fixed = tf.linalg.cholesky(K_ZG_fixed)
        factor_ZG_G_fixed = tf.linalg.triangular_solve(L_ZG_fixed, K_ZG_G_fixed, lower=True)
        K_ZG_inv_K_ZG_G_fixed = tf.linalg.triangular_solve(tf.transpose(L_ZG_fixed),
                                                           factor_ZG_G_fixed,
                                                           lower=False)
        fG_mean_fixed = tf.matmul(K_ZG_inv_K_ZG_G_fixed, self.m_uG_fixed, transpose_a=True) # [b,1]
        fG_mean_fixed = tf.repeat(fG_mean_fixed, a, axis=1) # [b,a]
        fG_mean_fixed = vec(fG_mean_fixed) # [b,a] --> #[a*b,1]
 
        # Fixed Effect Covariance (X)
        K_X_fixed = self.KX_fixed(X, X) # [a,a]
        S_X_fixed = tf.matmul(self.L_uX_fixed, self.L_uX_fixed, transpose_b=True) # [mX,mX]

        fX_cov_fixed = K_X_fixed
        fX_cov_fixed += tf.matmul(tf.matmul(K_ZX_inv_K_ZX_X_fixed, S_X_fixed, transpose_a=True),
                                  K_ZX_inv_K_ZX_X_fixed)
        fX_cov_fixed -= tf.matmul(K_ZX_inv_K_ZX_X_fixed, K_ZX_X_fixed, transpose_a=True) # [a,a]

        # Fixed Effect Covariance (G)
        K_G_fixed = self.KG_fixed(G, G) # [b,b]
        S_G_fixed = tf.matmul(self.L_uG_fixed, self.L_uG_fixed, transpose_b=True) # [mG,mG]

        fG_cov_fixed = K_G_fixed
        fG_cov_fixed += tf.matmul(tf.matmul(K_ZG_inv_K_ZG_G_fixed, S_G_fixed, transpose_a=True),
                                  K_ZG_inv_K_ZG_G_fixed)
        fG_cov_fixed -= tf.matmul(K_ZG_inv_K_ZG_G_fixed, K_ZG_G_fixed, transpose_a=True) # [b,b]

        # Random Effect Mean and Covariance (X)
        K_X_random = self.KX_random(X, X) # [a,a]
        K_ZX_X_random = self.KX_random(self.ZX_random.Z, X) # [mX,a]
        K_ZX_random = self.KX_random(self.ZX_random.Z, self.ZX_random.Z)\
                      + jitter * tf.eye(self.mX_random, dtype=default_float()) # [mX,mX]
        L_ZX_random = tf.linalg.cholesky(K_ZX_random)
        factor_ZX_X_random = tf.linalg.triangular_solve(L_ZX_random, K_ZX_X_random, lower=True)
        K_ZX_inv_K_ZX_X_random = tf.linalg.triangular_solve(tf.transpose(L_ZX_random),
                                                            factor_ZX_X_random,
                                                            lower=False)

        OH_bp = dmgp.core.OH(G_ids, self.G_ids) # [b,p], contains zero-vector if there is no match
        mask_X = tf.linalg.diag(tf.reduce_sum(OH_bp, axis=1)) # [b,b]

        m_uX_random = tf.gather(self.m_uX_random, tf.math.argmax(OH_bp, axis=1), axis=0) # [p,mX,1] --> [b,mX,1]
        m_uX_random = tf.matmul(mask_X, tf.squeeze(m_uX_random, axis=-1))[...,tf.newaxis] # [b,mX,1]
        fX_mean_random = tf.map_fn(compute_mean_X_k, m_uX_random) # [b,mX,1] --> [b,a,1]
        fX_mean_random = vec(tf.squeeze(fX_mean_random, axis=-1)) # [b,a,1] --> [a*b,1]

        L_uX_random = tf.gather(self.L_uX_random, tf.math.argmax(OH_bp, axis=1), axis=0) # [p,mX,mX] --> [b,mX,mX]
        L_uX_random = tf.reshape(tf.matmul(mask_X, tf.reshape(L_uX_random, (b,-1))), # [b,mX**2]
                                 (b,self.mX_random,self.mX_random))
        S_X_random = tf.matmul(L_uX_random, tf.transpose(L_uX_random, perm=[0,2,1])) # [b,mX,mX]
        fX_cov_random = K_X_random # [a,a]
        fX_cov_random += tf.map_fn(compute_cov_X_k, S_X_random) # [b,a,a], broadcast sum
        fX_cov_random -= tf.matmul(K_ZX_inv_K_ZX_X_random, K_ZX_X_random, transpose_a=True) # [b,a,a]

        # Mean
        tf.debugging.assert_shapes([
            (fX_mean_fixed, ('n','q')),
            (fX_mean_random, ('n','q')),
            (fG_mean_fixed, ('n','q'))
        ])

        fX_mean = fX_mean_fixed + fX_mean_random
        fG_mean = fG_mean_fixed
        f_mean = fX_mean + fG_mean

        if full_cov:
            # Fixed effect covariance
            fX_cov_fixed = KP(fX_cov_fixed, tf.ones((b,b), dtype=default_float())) # [a*b,a*b]
            fG_cov_fixed = KP(tf.ones((a,a), dtype=default_float()), fG_cov_fixed) # [a*b,a*b]

            # Random effect covariance
            fX_cov_random = tf.unstack(fX_cov_random) # List of b [a,a] tensors
            fX_cov_random = [tf.linalg.LinearOperatorFullMatrix(cov_k) for cov_k in fX_cov_random]
            fX_cov_random = tf.linalg.LinearOperatorBlockDiag(fX_cov_random).to_dense() # [b*a,b*a]

            K1_idx = commutation_mat(b,a) # Reordering
            fX_cov_random = tf.gather(fX_cov_random, K1_idx, axis=0)
            fX_cov_random = tf.gather(fX_cov_random, K1_idx, axis=1) # [a*b,a*b]

            assert(fX_cov_fixed.shape == fX_cov_random.shape)
            assert(fX_cov_fixed.shape == fG_cov_fixed.shape)
            
            fX_cov = fX_cov_fixed + fX_cov_random
            fG_cov = fG_cov_fixed
            f_cov = fX_cov + fG_cov

            if sep == 'fixed':
                return ((fX_mean_fixed, fX_mean_random, fG_mean_fixed), 
                        (fX_cov_fixed, fX_cov_random, fG_cov_fixed))
            else:
                return (f_mean, f_cov)

        else:
            # Fixed effect variance
            fX_var_fixed = tf.linalg.diag_part(fX_cov_fixed)[:,tf.newaxis] # [a,1]
            fX_var_fixed = KP(fX_var_fixed, tf.ones((b,1), dtype=default_float())) # [a*b,1]

            fG_var_fixed = tf.linalg.diag_part(fG_cov_fixed)[:,tf.newaxis] # [b,1]
            fG_var_fixed = KP(tf.ones((a,1), dtype=default_float()), fG_var_fixed) # [a*b,1]

            # Random effect variance
            fX_var_random = tf.linalg.diag_part(fX_cov_random) # [b,a]
            fX_var_random = vec(fX_var_random) # [a*b,1]

            tf.debugging.assert_shapes([
                (fX_var_fixed, ('n','q')),
                (fX_var_random, ('n','q')),
                (fG_var_fixed, ('n','q'))
            ])

            fX_var = fX_var_fixed + fX_var_random
            fG_var = fG_var_fixed
            f_var = fX_var + fG_var

            if sep == 'fixed':
                return ((fX_mean_fixed, fX_mean_random, fG_mean_fixed), 
                        (fX_var_fixed, fX_var_random, fG_var_fixed))
            else:
                return (f_mean, f_var)

    # Returns predicted y based on posterior mean and covariance
    def predict_y(self, data, ids):
        f_mean, f_var = self.predict_f(data, ids)

        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    # Returns log of the predictive density for given data
    def predict_log_density(self, data, ids):
        _, _, Y = data
        f_mean, f_var = self.predict_f(data, ids)
        
        return self.likelihood.predict_log_density(tf.squeeze(f_mean), 
                                                   tf.squeeze(f_var), 
                                                   tf.squeeze(vec(Y)))

# Mixed-Effects Gaussian Process (MGP)
class MGP(GPModel):
    def __init__(self, shape, ids, kernel, ll, ind_points, initial_cov='prior'):
        super().__init__(kernel=kernel, likelihood=ll, num_latent_gps=1)

        self.p, self.n = shape
        self.G_ids, self.X_ids = ids
        
        # Inducing points
        self.Z_fixed = inducingpoint_wrapper(ind_points[0])
        self.Z_random = inducingpoint_wrapper(ind_points[1])
        
        self.m_fixed = ind_points[0].num_inducing
        self.m_random = ind_points[1].num_inducing
        self.n_ind_points = {'Z_fixed': self.m_fixed,
                             'Z_random': self.m_random}
        
        # Kernel functions
        self.KX_fixed = copy.deepcopy(kernel) # Prior Fixed Effect Kernel
        self.KX_random = copy.deepcopy(kernel) # Prior Random Effect Kernel
        
        # Disable training for unused parameters
        gpflow.set_trainable(self.kernel.parameters[0], False)
        gpflow.set_trainable(self.kernel.parameters[1], False)

        # Variational parameters
        if initial_cov == 'prior':
            self.L_uX_fixed = Parameter(self.prior_cov(self.KX_fixed, self.Z_fixed), transform=triangular()) # [m,m]
            self.L_uX_random = Parameter(tf.repeat(self.prior_cov(self.KX_random, self.Z_random)[tf.newaxis,...], self.p, axis=0),
                                         transform=triangular()) # [p,m,m]

        elif initial_cov == 'identity':
            self.L_uX_fixed = Parameter(np.eye(self.m_fixed, dtype=default_float()), transform=triangular())
            self.L_uX_random = Parameter(tf.repeat(np.eye(self.m_random, dtype=default_float())[tf.newaxis,...], self.p, axis=0),
                                         transform=triangular()) # [p,m,m]

        else:
            raise RuntimeError(f'[Error] Invalid covariance initialization: {initial_cov}')

        self.m_uX_fixed = Parameter(np.zeros((self.m_fixed,1), dtype=default_float())) # [m,1]
        self.m_uX_random = Parameter(np.zeros((self.p,self.m_random,1), dtype=default_float())) # [p,m,1]

    # Computes the prior covariance for initialization
    def prior_cov(self, K, Z, jitter=default_jitter()):
        cov = K(Z.Z, Z.Z)
        cov += jitter * tf.eye(Z.Z.shape[0], dtype=default_float())

        return tf.linalg.cholesky(cov)

    # KL divergence term for fixed effect inducing points
    def fixed_prior_kl(self, jitter=default_jitter()):
        K = self.KX_fixed(self.Z_fixed.Z, self.Z_fixed.Z)\
            + jitter * tf.eye(self.m_fixed, dtype=default_float())

        return kullback_leiblers.gauss_kl(
            self.m_uX_fixed, self.L_uX_fixed[tf.newaxis,:], K=K
        )

    # Total KL divergence
    def total_prior_kl(self, jitter=default_jitter()):
        # Fixed KL
        kl = self.fixed_prior_kl()

        # Random KL
        K = self.KX_random(self.Z_random.Z, self.Z_random.Z)\
            + jitter * tf.eye(self.m_random, dtype=default_float())
        kl += kullback_leiblers.gauss_kl(tf.transpose(tf.squeeze(self.m_uX_random, axis=-1)),
                                         self.L_uX_random, K=K)

        return kl
    
    # Loss function
    def training_loss(self, data, ids):
        return -self.maximum_log_likelihood_objective(data, ids)

    # ELBO objective
    def maximum_log_likelihood_objective(self, data, ids):
        return self.elbo(data, ids)

    # Computes the ELBO
    # Note: Expects mini-batch sampling to be handled externally
    def elbo(self, data, ids):
        # Unpack data-tuple
        X, Y = data
        b, a = Y.shape
        
        # Compute the parameters of q(f)
        f_mean, f_var = self.predict_f(data, ids)
        
        # Expected log-likelihood
        var_exp = self.likelihood.variational_expectations(tf.squeeze(f_mean), 
                                                           tf.squeeze(f_var), 
                                                           tf.squeeze(vec(Y)))
        
        # Compute KL term
        kl = self.total_prior_kl()
        
        # Scale
        scale = (self.n * self.p) / (a * b)
        
        return tf.reduce_sum(var_exp) * scale - kl

    # Returns the mean and marginal variance of f for mini-batch/new data
    def predict_f(self, data, ids, sep=None, full_cov=False, full_output_cov=False):
        # Unpack data-tuple
        X, Y = data
        b, a = Y.shape

        G_ids, X_ids = ids

        # Commutation matrix for A KP B where A is [m,m] and B is [n,n]
        # Computes K(n,m), for K(n,m)(A KP B)K(m,n) = B KP A
        @tf.function
        def commutation_mat(m,n):
            K1_idx = tf.reshape(tf.transpose(tf.stack(tf.split(tf.range(m*n), n))), (-1,))
            
            return K1_idx

        # Helper functions for computing random effects mean and covariance
        def compute_mean_k(m_k):
            return tf.matmul(K_Z_inv_K_ZX_random, m_k, transpose_a=True)

        def compute_cov_k(S_k):
            return tf.matmul(tf.matmul(K_Z_inv_K_ZX_random, S_k, transpose_a=True), 
                             K_Z_inv_K_ZX_random)

        # Fixed Effect Mean
        K_ZX_fixed = self.KX_fixed(self.Z_fixed.Z, X)
        K_Z_fixed = self.KX_fixed(self.Z_fixed.Z, self.Z_fixed.Z)\
                    + default_jitter() * tf.eye(self.m_fixed, dtype=default_float())
        L_Z_fixed = tf.linalg.cholesky(K_Z_fixed)
        factor_ZX_fixed = tf.linalg.triangular_solve(L_Z_fixed, K_ZX_fixed, lower=True)
        K_Z_inv_K_ZX_fixed = tf.linalg.triangular_solve(tf.transpose(L_Z_fixed),
                                                        factor_ZX_fixed,
                                                        lower=False)
        f_mean_fixed = tf.matmul(K_Z_inv_K_ZX_fixed, self.m_uX_fixed, transpose_a=True) # [a,1]
        f_mean_fixed = tf.repeat(f_mean_fixed[tf.newaxis,:], b, axis=0) # [b,a,1]
        f_mean_fixed = vec(tf.squeeze(f_mean_fixed, axis=-1)) # [b,a] --> [a*b,1]
        
        # Fixed Effect Covariance
        K_X_fixed = self.KX_fixed(X, X) # [a,a]
        S_fixed = tf.matmul(self.L_uX_fixed, self.L_uX_fixed, transpose_b=True) # [m,m]

        f_cov_fixed = K_X_fixed
        f_cov_fixed += tf.matmul(tf.matmul(K_Z_inv_K_ZX_fixed, S_fixed, transpose_a=True),
                                 K_Z_inv_K_ZX_fixed)
        f_cov_fixed -= tf.matmul(K_Z_inv_K_ZX_fixed, K_ZX_fixed, transpose_a=True) # [a,a]

        # Random Effects
        K_X_random = self.KX_random(X, X) # [a,a]
        K_ZX_random = self.KX_random(self.Z_random.Z, X) # [m,a]
        K_Z_random = self.KX_random(self.Z_random.Z, self.Z_random.Z)\
                     + default_jitter() * tf.eye(self.m_random, dtype=default_float()) # [m,m]
        L_Z_random = tf.linalg.cholesky(K_Z_random)
        factor_ZX_random = tf.linalg.triangular_solve(L_Z_random, K_ZX_random, lower=True)
        K_Z_inv_K_ZX_random = tf.linalg.triangular_solve(tf.transpose(L_Z_random),
                                                         factor_ZX_random,
                                                         lower=False)

        OH_bp = dmgp.core.OH(G_ids, self.G_ids) # [b,p], contains zero-vector if there is no match
        mask = tf.linalg.diag(tf.reduce_sum(OH_bp, axis=1)) # [b,b]
        
        m_uX_random = tf.gather(self.m_uX_random, tf.math.argmax(OH_bp, axis=1), axis=0) # [p,m,1] --> [b,m,1]
        m_uX_random = tf.matmul(mask, tf.squeeze(m_uX_random, axis=-1))[...,tf.newaxis] # [b,m,1]
        f_mean_random = tf.map_fn(compute_mean_k, m_uX_random) # [b,m,1] --> [b,a,1]
        f_mean_random = vec(tf.squeeze(f_mean_random, axis=-1)) # [b,a,1] --> [a*b,1]

        L_uX_random = tf.gather(self.L_uX_random, tf.math.argmax(OH_bp, axis=1), axis=0) # [p,m,m] --> [b,m,m]
        L_uX_random = tf.reshape(tf.matmul(mask, tf.reshape(L_uX_random, (b,-1))), # [b,m**2]
                                 (b,self.m_random,self.m_random))
        S_random = tf.matmul(L_uX_random, tf.transpose(L_uX_random, perm=[0,2,1])) # [b,m,m]
        f_cov_random = K_X_random # [a,a]
        f_cov_random += tf.map_fn(compute_cov_k, S_random) # [b,a,a], broadcast sum
        f_cov_random -= tf.matmul(K_Z_inv_K_ZX_random, K_ZX_random, transpose_a=True) # [b,a,a]

        # Mean
        f_mean = f_mean_fixed + f_mean_random

        if full_cov:
            # Fixed effect covariance
            f_cov_fixed = KP(f_cov_fixed, tf.ones((b,b), dtype=default_float())) # [a*b,a*b]

            # Random effect covariance
            f_cov_random = tf.unstack(f_cov_random) # List of p [a,a] tensors
            f_cov_random = [tf.linalg.LinearOperatorFullMatrix(cov_k) for cov_k in f_cov_random]
            f_cov_random = tf.linalg.LinearOperatorBlockDiag(f_cov_random).to_dense()

            K1_idx = commutation_mat(b,a) # Reordering
            f_cov_random = tf.gather(f_cov_random, K1_idx, axis=0)
            f_cov_random = tf.gather(f_cov_random, K1_idx, axis=1)

            assert(f_cov_fixed.shape == f_cov_random.shape)
            f_cov = f_cov_fixed + f_cov_random

            if sep == 'fixed':
                return ((f_mean_fixed, f_mean_random), (f_cov_fixed, f_cov_random))
            else:
                return (f_mean, f_cov)

        else:
            # Fixed effect variance
            f_var_fixed = tf.linalg.diag_part(f_cov_fixed)[:,tf.newaxis] # [a,1]
            f_var_fixed = KP(f_var_fixed, tf.ones((b,1), dtype=default_float())) # [a*b,1]

            # Random effect variance
            f_var_random = tf.linalg.diag_part(f_cov_random) # [b,a]
            f_var_random = vec(f_var_random) # [a*b,1]

            assert(f_var_fixed.shape == f_var_random.shape)
            f_var = f_var_fixed + f_var_random

            if sep == 'fixed':
                return ((f_mean_fixed, f_mean_random), (f_var_fixed, f_var_random))
            else:
                return (f_mean, f_var)

    # Returns predicted y based on posterior mean and covariance
    def predict_y(self, data, ids):
        f_mean, f_var = self.predict_f(data, ids)

        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    # Returns log of the predictive density for given data
    def predict_log_density(self, data, ids):
        _, Y = data
        f_mean, f_var = self.predict_f(data, ids)
        
        return self.likelihood.predict_log_density(tf.squeeze(f_mean), 
                                                   tf.squeeze(f_var), 
                                                   tf.squeeze(vec(Y)))

