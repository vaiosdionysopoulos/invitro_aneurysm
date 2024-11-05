from typing import Callable
import blackjax.sgmcmc.diffusions as diffusions
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random, vmap, lax, grad
import blackjax
import blackjax.sgmcmc.sghmc as sghmc
import equinox as eqx
import os
from blackjax.base import SamplingAlgorithm
from config import *

rng_key=jax.random.PRNGKey(0)
data_spatial_points,data_time_values,data_mag_values,data_phase_values,sigma_mag,sigma_phase_x,sigma_phase_y,sigma_phase_z,num_obs=load_data()
dataset=CombinedTimeStepDataset(data_spatial_points, data_mag_values, data_phase_values, data_time_values)





def energy_conserv_hmc_sampling(theta_init,
                batch_size,
                total_samples, thinning_factor, lr_package,
                rng_key,temperature=1.0,num_integration_steps=10):
    
    a1, a2, c = lr_package
  
    sghmc = blackjax.sghmc(grad_estimator,num_integration_steps)

    def sghmc_step(rng_key,position,batch,step_size,temperature=1.0):

        position_proposed=sghmc.step(rng_key, position, batch, step_size,temperature)

        # Compute log-probabilities for MH acceptance ratio
        logprob_current = logprob_fn(position,batch)
        logprob_proposed = logprob_fn(position_proposed,batch)
        
        # Calculate MH acceptance ratio
        acceptance_ratio = jnp.exp(logprob_proposed - logprob_current)
        
        # Accept or reject the proposed position
        rng_key, subkey = jax.random.split(rng_key)
        u = jax.random.uniform(subkey)
        
        accepted=u<acceptance_ratio
        new_state = lax.cond(accepted, lambda _: position_proposed, lambda _: position, operand=None)
        
        return new_state

    sghmc_step=eqx.filter_jit(sghmc_step)

    #Executes one step of the SGHMC algirithm 
    @eqx.filter_jit
    def sghmc_step_fun(iter,carry):
        position, rng_key = carry
        rng_key, batch_key, sample_key = jax.random.split(rng_key, 3)

        data_batch=get_batch(dataset,batch_size,batch_key)
        
        lr = lr_schedule(iter, a1, a2, c)
        new_position=sghmc_step(sample_key, position, data_batch, lr,temperature)
        return (new_position, rng_key)
    
    #Executes thinning_factor iterations and returns the last sample
    def last_sample(current_iter,init):
        val=jax.lax.fori_loop(current_iter,int(current_iter+thinning_factor),sghmc_step_fun,init)
        sample,rng_key=val
        return current_iter+thinning_factor,sample,rng_key
    
    
    current_iter = 0
    init = (theta_init, rng_key)
    if thinning_factor==1:
        if not os.path.exists("SGHMC/models_no_thinning"):
            os.makedirs("SGHMC/models_no_thinning")
    else:
        if not os.path.exists("SGHMC/models"):
                os.makedirs("SGHMC/models_no_thinning")

    order=1
    no_thinning=thinning_factor==1
    
    for i in range(total_samples):
        iter, new_sample, rng_key = last_sample(current_iter, init)
        if no_thinning:
            eqx.tree_serialise_leaves(f"SGHMC/models_no_thinning/SGHMC_sampling_{order}_batch_size_{batch_size}_thinning_{thinning_factor}.eqx",new_sample)  # Directly set the dictionary in the preallocated list
        else:
            eqx.tree_serialise_leaves(f"SGHMC/models/SGHMC_sampling_{order}_batch_size_{batch_size}_thinning_{thinning_factor}.eqx",new_sample)
        current_iter = iter
        init = (new_sample, rng_key)
        order+=1

    

#We define the learning rate package that dicatetes the learning rate at each step 
#and the initial position. User may chaing name of file (make sure the saved pytree has 
#the same structure as the initial model, meaning params_init), as well as the other parametres
#of the sampling.
def main():
    lr_package = (0.01, 0.51, 0.001)
    theta_init=eqx.filter(params_init,eqx.is_array)
    batch_size=int(0.05*num_obs)
    total_samples=10   
    thinning=1
    energy_conserv_hmc_sampling(theta_init,batch_size,total_samples,thinning,lr_package,rng_key)

    

if __name__=="__main__":
    main()
   
        

        