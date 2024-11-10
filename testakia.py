from typing import Callable
import numpy as np
import pickle
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random, vmap, lax, grad
from torch.utils.data import Dataset
from jifty import FourierFeaturesLayer, MLPModified
import blackjax
import blackjax.sgmcmc.gradients as gradients
import functools as ft
from functools import partial
import equinox as eqx
from config import *

rng_key=jax.random.PRNGKey(0)
data_spatial_points,data_time_values,data_mag_values,data_phase_values,sigma_mag,sigma_phase_x,sigma_phase_y,sigma_phase_z,num_obs=load_data()
dataset=CombinedTimeStepDataset(data_spatial_points, data_mag_values, data_phase_values, data_time_values)
params_init=model_init(rng_key)
min_vals = data_spatial_points.min(axis=0)
max_vals = data_spatial_points.max(axis=0)
x_range=[min_vals[0], max_vals[0]]
y_range=[min_vals[1], max_vals[1]]
z_range=[min_vals[2], max_vals[2]]
t_range=[data_time_values.min(), data_time_values.max()]




def model_funcs(params):
    geom = lambda x : params["nn_geom"](x)[0]
    velx = lambda x : params["nn_vel_v_x"](x)[0]
    vely = lambda x : params["nn_vel_v_y"](x)[0]
    velz = lambda x : params["nn_vel_v_z"](x)[0]
    
    return geom,velx,vely,velz


def select_networks(d):
    if isinstance(d, dict) and all(key in d for key in ["nn_geom", "nn_vel_v_x", "nn_vel_y", "nn_vel_v_z", "nn_press"]):
        return (d["nn_geom"], d["nn_vel_v_x"], d["nn_vel_y"], d["nn_vel_v_z"], d["nn_press"])
    return d  # return unchanged for non-dictionary it

@partial(jax.vmap, in_axes=(0,None))
def eval(x,funcs):
    geom,velx,vely,velz,pressure=funcs
    return geom(x),velx(x),vely(x),velz(x),pressure(x)










#This is working for iteratively stacking models together.
def combine_trees(tree1,tree2,tree1_iscombined):
    def add_leading_dim(leaf):
        return jnp.expand_dims(leaf, axis=0)
    def combine_leaf(leaf1,leaf2):
        return jnp.concatenate((leaf1,leaf2),axis=0)
    if tree1_iscombined:
        tree1_reshaped=tree1
    else:
        tree1_reshaped=jax.tree_util.tree_map(add_leading_dim,tree1)

    tree2_reshaped=jax.tree_util.tree_map(add_leading_dim,tree2)

    return jax.tree_util.tree_map(combine_leaf,tree1_reshaped,tree2_reshaped)


params_combined=combine_trees(params_init,params_init,False)
test=combine_trees(params_combined,params_init,True)


def eval_single_model(x, params):
    geom = lambda x : params["nn_geom"](x)
    velx = lambda x : params["nn_vel_v_x"](x)[0]
    vely = lambda x : params["nn_vel_v_y"](x)[0]
    velz = lambda x : params["nn_vel_v_z"](x)[0]
    
    return jnp.array([geom(x)[0],geom(x)[1],velx(x),vely(x),velz(x)])

@eqx.filter_vmap(in_axes = (eqx.if_array(0), None))
def eval_ensemble(model,x):
    return eval_single_model(x,model)

posterior_predictive_samples = vmap(eval_ensemble, (None,0), 1)


x=dataset[:3][0]
result=posterior_predictive_samples(test,x)

#result is a tensor of the form (models,points,evaluations)

def print_structure(x):
    print(type(x), x.shape if hasattr(x, 'shape') else x)
jax.tree_util.tree_map(print_structure,tree2)

