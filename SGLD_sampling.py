"""Implelemnts the SGLG algorithm for the parameters of our model. 
It keeps the samples specified by the varibale thinning_factor, and saved them as one single dictionary 
using Equinox serialisation functionalities. It is meant to work with trained parameters 
as a starting point, thus with no burn-in period"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random, vmap, lax, grad
from jifty import FourierFeaturesLayer, MLPModified
import blackjax
import blackjax.sgmcmc.gradients as gradients
import functools as ft
from functools import partial
import equinox as eqx
import pickle
import os
from config import *

dataset=CombinedTimeStepDataset(data_spatial_points, data_mag_values, data_phase_values, data_time_values)

#sampling function takes SGLG samples every thinning_factor iterations
def sgld_sampling(theta_init,
                batch_size,
                total_samples, thinning_factor, lr_package,
                rng_key,temperature=1.0):
    
    a1, a2, c = lr_package
  
    sgld = blackjax.sgld(grad_estimator)
    sgld_step = eqx.filter_jit(sgld.step)
    position = sgld.init(theta_init)

    #Executes one step of the SGLD algirithm 
    @eqx.filter_jit
    def sgld_step_fun(iter,carry):
        """One step of SGLD"""
        position, rng_key = carry
        rng_key, batch_key, sample_key = jax.random.split(rng_key, 3)

        data_batch=get_batch(dataset,batch_size,batch_key)
        
        lr = lr_schedule(iter, a1, a2, c)
        new_position=sgld_step(sample_key, position, data_batch, lr,temperature)
        return (new_position, rng_key)
    
    #Executes thinning_factor iterations and returns the last sample
    def last_sample(current_iter,init):
        val=jax.lax.fori_loop(current_iter,int(current_iter+thinning_factor),sgld_step_fun,init)
        sample,rng_key=val
        return current_iter+thinning_factor,sample,rng_key
    
    
    current_iter = 0
    init = (position, rng_key)
    if thinning_factor==1:
        if not os.path.exists("models_no_thinning_trained_params"):
            os.makedirs("models_no_thinning_trained_params")
    else:
        if not os.path.exists("models"):
                os.makedirs("models")

    order=1
    no_thinning=thinning_factor==1
    
    for i in range(total_samples):
        iter, new_sample, rng_key = last_sample(current_iter, init)
        if no_thinning:
            eqx.tree_serialise_leaves(f"models_no_thinning_trained_params/SGLD_sampling_{order}_batch_size_{batch_size}_thinning_{thinning_factor}.eqx",new_sample)  # Directly set the dictionary in the preallocated list
        else:
            eqx.tree_serialise_leaves(f"models/SGLD_sampling_{order}_batch_size_{batch_size}_thinning_{thinning_factor}.eqx",new_sample)
        current_iter = iter
        init = (new_sample, rng_key)
        order+=1

    

#We define the learning rate package that dicatetes the learning rate at each step 
#and the initial position. User may chaing name of file (make sure the saved pytree has 
#the same structure as the initial model, meaning params_init), as well as the other parametres
#of the sampling.
def main():
    lr_package = (0.01, 0.51, 0.001)
    params=eqx.tree_deserialise_leaves("runs/params_trained_bs_34830_num_classes_2.eqx",params_init)
    theta_init=eqx.filter(params,eqx.is_array)
    batch_size=int(0.05*num_obs)
    total_samples=10000
    thinning=1
    sgld_sampling(theta_init,batch_size,total_samples,thinning,lr_package,rng_key)

    

if __name__=="__main__":
    main()
   

