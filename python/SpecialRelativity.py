import os, time
import jax
import jax.numpy as jnp
import numpy as np
import argparse
from jax import jit
from jax.experimental.ode import odeint
from functools import partial
try: from jax.experimental import stax, optimizers
except: from jax.example_libraries import stax, optimizers # for older versions of python=3.7
from HyperparameterSearch import extended_mlp
from models import mlp as make_mlp
from utils import wrap_coords

def lagrangian_eom(lagrangian, state, conditionals, t=None):
    q, q_t = jnp.split(state, 2)
    q = q / 10.0 #Normalize
    conditionals = conditionals / 10.0
    q_t = q_t
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t, conditionals))
          @ (jax.grad(lagrangian, 0)(q, q_t, conditionals)
             - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t, conditionals) @ q_t))
    return jnp.concatenate([q_t, q_tt])

# replace the lagrangian with a parameteric model
def learned_dynamics(params, nn_forward_fn):
    @jit
    def dynamics(q, q_t, conditionals):
    #     assert q.shape == (2,)
        state = jnp.concatenate([q, q_t, conditionals])
        return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
    return dynamics
