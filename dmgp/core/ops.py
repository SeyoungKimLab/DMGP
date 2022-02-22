import sys 
sys.path.append('../../') # Add DMGP root

import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.linalg import (
    LinearOperatorFullMatrix,
    LinearOperatorKronecker,
    LinearOperatorBlockDiag
)

import gpflow
from gpflow.models.util import inducingpoint_wrapper
from gpflow.inducing_variables import InducingPoints
from gpflow.covariances import Kuu, Kuf
from gpflow.config import default_float, default_jitter
from gpflow.utilities import to_default_float

gpflow.config.set_default_jitter(1e-6)

# Vectorization operation: vec(A)
def vec(A):
    return tf.reshape(tf.transpose(A), (-1,1))

# Inverse vectorization operation: vec_inv(A)
def vec_inv(A, shape):
    p, n = shape

    return np.reshape(A, (n,p)).T

# Inverse vectorization operation: vec_inv_tf(A) (TF)
def vec_inv_tf(A, shape):
    p, n = shape

    return tf.transpose(tf.reshape(A, (n,p)))

# Inverse operation: A^{-1}
# Adds diagonal noise to ensure invertibility, default_jitter=1e-6
def inv(A, jitter=default_jitter()):
    n = A.shape[0]
    
    return tf.linalg.inv(A + jitter * tf.eye(n, dtype=default_float()))

def logdet(A, jitter=default_jitter()):
    n = A.shape[0]

    return tf.linalg.logdet(A + jitter * tf.eye(n, dtype=default_float()))

# Kronecker product operation: A KP B (tensordot)
# Note: Faster than other methods for KP calculation
def KP(A, B):
    n1, n2 = A.shape
    m1, m2 = B.shape
    kp = tf.tensordot(A, B, axes=0)
    kp = tf.transpose(kp, (0,2,1,3))
    kp = tf.reshape(kp, (n1*m1, n2*m2))

    return kp

# Kronecker product operation: A KP B (LinearOperator)
def KP_LinearOperator(A, B):
    op1 = LinearOperatorFullMatrix(A)
    op2 = LinearOperatorFullMatrix(B)
    kp = LinearOperatorKronecker([op1, op2]).to_dense()

    return kp

# One-hot vectorization comparing two arrays of indices
def OH(idxs1, idxs2):
    return tf.cast(idxs1[:,None] == idxs2, tf.float64)