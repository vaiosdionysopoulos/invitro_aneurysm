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
import gc
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

#Ranges of each parameter 
lower_range=[x_range[0], y_range[0], z_range[0], t_range[0]]
upper_range=[x_range[1], y_range[1], z_range[1], t_range[1]]

#sampling function takes SGLG samples every thinning_factor iterations
def sgld_sampling(theta_init ,
                batch_size, batch_size_physics, reference_point,
                total_samples, thinning_factor, lr_package,
                rng_key,save_directory,temperature=1.0):
    
    a1, a2, c = lr_package
  
    sgld = blackjax.sgld(grad_estimator)
    sgld_step = eqx.filter_jit(sgld.step)
    position = sgld.init(theta_init)

    #Executes one step of the SGLD algirithm 
    @eqx.filter_jit
    def sgld_step_fun(iter,carry):
        """One step of SGLD"""
        position, rng_key = carry
        rng_key, batch_key, sample_key, physics_key = jax.random.split(rng_key, 4)

        #Physics batch should have the form (num_points,4)
        batch_data=get_batch(dataset,batch_size,batch_key)
        batch_physics=collocation_grid(batch_size_physics,physics_key,lower_range,upper_range)


        batch=[batch_data, batch_physics, reference_point]
        

        lr = lr_schedule(iter, a1, a2, c)
        new_position=sgld_step(sample_key, position, batch, lr,temperature)
        return (new_position, rng_key)
    
    #Executes thinning_factor iterations and returns the last sample
    def last_sample(current_iter,init):
        val=jax.lax.fori_loop(current_iter,int(current_iter+thinning_factor),sgld_step_fun,init)
        sample,rng_key=val
        return current_iter+thinning_factor,sample,rng_key
    
    
    current_iter = 0
    init = (position, rng_key)
    combined_samples= None
    
    
    for i in range(total_samples):
        iter, new_sample, rng_key = last_sample(current_iter, init)
        if i==0:
            combined_samples=new_sample
        elif i==1:
            combined_samples=combine_trees(combined_samples,new_sample,False)

        else:
            combined_samples=combine_trees(combined_samples,new_sample,True)
    
        if i==199:
            filename=f"SGLD_samples_{(i+1)-200}-{i+1}_batch_size_{batch_size}_thinning_{thinning_factor}.eqx"
            filepath = os.path.join(save_directory, filename)
            eqx.tree_serialise_leaves(filepath,combined_samples)
            del combined_samples
            gc.collect()
        elif i==total_samples-1:
            filename=f"SGLD_samples_{0}-{i+1}_batch_size_{batch_size}_thinning_{thinning_factor}.eqx"
            filepath = os.path.join(save_directory, filename)
            eqx.tree_serialise_leaves(filepath,combined_samples)
            del combined_samples
            gc.collect()

        current_iter = iter
        init = (new_sample, rng_key)
        

    

#We define the learning rate package that dictates the learning rate at each step 
#and the initial position. User may change name of file.
def main():
    lr_package = (0.01, 0.51, 0.001)
    theta_init=eqx.filter(params_init,eqx.is_array)
    reference_point=jnp.array(lower_range)
    batch_size=200
    batch_size_physics=128
    total_samples=30
    thinning=1
    sgld_sampling(theta_init,batch_size,batch_size_physics, reference_point, total_samples,thinning,lr_package,rng_key,"test")

    

if __name__=="__main__":
    main()
   

