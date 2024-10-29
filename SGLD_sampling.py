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
from SGLD_prototype import CombinedTimeStepDataset,get_batch,lr_schedule,loss_data,l2_norm,params_init

rng_key=jax.random.PRNGKey(1)
# # Data
with open("data_mri.pkl", 'rb') as f:
    data = pickle.load(f)
    
#I HAVE TO MAKE EVRYTHING JAX IN THE BEGINNING NOT IN THE STEP_SCAN

data_spatial_points = data["spatial_points"]
data_time_values =data["time_values"]
data_mag_values = data["mag_values"]
data_phase_values = data["phase_values"]

sigma_mag = 0.002
sigma_phase_x = 0.002
sigma_phase_y = 0.002
sigma_phase_z = 0.002

x_size = data["nx"]
y_size = data["ny"]
z_size = data["nz"]
nt = data["nt"]

constant_num_points = data_spatial_points.shape[0]
constant_num_timesteps =data_time_values.shape[0]
num_obs=constant_num_points*constant_num_timesteps


#sampling function takes SGLG samples every thinning_factor iterations
def sgld_sampling(theta_init,
                batch_size,
                total_samples, thinning_factor, lr_package,
                rng_key,temperature=1.0):
    
    a1, a2, c = lr_package
    dataset=CombinedTimeStepDataset(data_spatial_points, data_mag_values, data_phase_values, data_time_values)
    # log posterior model 
    @eqx.filter_jit
    def logprob_fn(position, batch):
        # blackjax calls the parameters which get updated 'position'
        # batch are objects that can be passed between iterations

        # log prior of DNN parameters
        # we are using a prior of N(0,1) for these, so we just need log normal

        # you can access the actual NN parameter values like this
        x=batch[0]
        y_mag=batch[1]
        y_phase=batch[2]
        l2_total = (
        l2_norm(position["nn_geom"]) + 
        l2_norm(position["nn_vel_v_x"]) + 
        l2_norm(position["nn_vel_v_y"]) + 
        l2_norm(position["nn_vel_v_z"])
    )
        return  num_obs * jnp.mean(loss_data(position,x,y_mag,y_phase), axis = 0)+ 0.5e3 * l2_total
    

    def gradient_clip_threshold(iter_, max_clip=1, decay_factor=0.99):
        return max_clip * (decay_factor ** iter_)
    @eqx.filter_jit
    def grad_estimator(p, batch, clip_val=4.):
        """ Compute clipped estimate of gradient of log probability """
        # Call the JIT compiled gradient function
        _,g =eqx.filter_value_and_grad(logprob_fn)(p,batch)
        
        # Clipping the gradient
        return jtu.tree_map(ft.partial(jnp.clip, a_min=-clip_val, a_max=clip_val), g)

    # build the SGLD sampler now
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
        if not os.path.exists("models_no_thinning"):
            os.makedirs("models_no_thinning")
    else:
        if not os.path.exists("models"):
                os.makedirs("models")

    order=1
    no_thinning=thinning_factor==1
    
    for i in range(total_samples):
        iter, new_sample, rng_key = last_sample(current_iter, init)
        if no_thinning:
            eqx.tree_serialise_leaves(f"models_no_thinning/SGLD_sampling_{order}_batch_size_{batch_size}_thinning_{thinning_factor}.eqx",new_sample)  # Directly set the dictionary in the preallocated list
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
    theta_init= eqx.filter(params_init, eqx.is_array)
    batch_size=int(0.05*num_obs)
    total_samples=10000
    thinning=1
    sgld_sampling(theta_init,batch_size,total_samples,thinning,lr_package,rng_key)

    

if __name__=="__main__":
    main()
   

"""samples = [{} for _ in range(total_samples)]
    current_iter = 0
    init = (position, rng_key)

    # Calculates and stores the samples in the list of dictionaries "samples"
    for i in range(total_samples):
        iter, new_sample, rng_key = last_sample(current_iter, init)
        samples[i] = new_sample  # Directly set the dictionary in the preallocated list
        current_iter = iter
        init = (new_sample, rng_key)

    return samples

#We define the learning rate package that dicatetes the learning rate at each step 
#and the initial position. User may chaing name of file (make sure the saved pytree has 
#the same structure as the initial model, meaning params_init), as well as the other parametres
#of the sampling.
def main():
    lr_package = (0.01, 0.51, 0.001)
    trained_params=eqx.tree_deserialise_leaves("runs/params_trained_bs_100_num_classes_2.eqx",params_init)
    theta_init= eqx.filter(trained_params, eqx.is_array)
    batch_size=1000
    total_samples=200
    thinning=10
    mysamples=sgld_sampling(theta_init,batch_size,total_samples,thinning,lr_package,rng_key)

    num_models = len(mysamples)

    networks = {
        f"model_{i}": nn for i, nn in enumerate(mysamples)
    }


    eqx.tree_serialise_leaves(f"runs/SGLD_samples_{num_models}_batch_size_{batch_size}_thinning_{thinning}.eqx",networks)"""

