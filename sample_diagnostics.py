"""Executes diagnostics for MCMC samples of neural networks. To work, each sample needs to be stored
in its own file in a directory that doesn't contain other files."""
import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import blackjax.diagnostics as diagnostics 
import equinox as eqx
import os
import re 
from plots_MCMC import *
from config import *
from memory_profiler import profile
import gc
from SGLD_sampling import dataset

def extract_single_model_components(model):
    comp1=model["nn_geom"].layers[1].layers[0].v[2,189]
    comp2=model["nn_vel_v_x"].layers[1].layers[1].bias[243]
    comp3=model["nn_vel_v_y"].layers[1].layers[1].bias[6]
    comp4=model["nn_vel_v_z"].layers[1].layers[2].v[53,1]
    return jnp.array([comp1,comp2,comp3,comp4])

@profile 
def cumulative_r_hats(components):
    _, models, _ = components.shape  # Unpack the shape of the array

    cumulative_r_hats = []  # Preallocate with the correct shape

    for model_count in range(models):  # Iterate over the models
    # Increment model_count for non-empty slice
        cumulative_components = components[:, :model_count + 2, :]  # Shape (chains, model_count + 1, components)

    # Calculate R-hat for the current cumulative set of models
        r_hat = diagnostics.potential_scale_reduction(cumulative_components)
    
    # Ensure r_hat can be correctly stored in cumulative_r_hats
        cumulative_r_hats.append(r_hat)  # Store the result in the preallocated array

    return cumulative_r_hats


@profile
def cumulative_ess(components):
    _, models,_ = components.shape  # Unpack the shape of the array
    
    cumulative_ess =[]  # Preallocate with the correct shape

    for model_count in range(models):  # Iterate over the models
        # Increment model_count for non-empty slice
        cumulative_components = components[:, :model_count + 2, :]  # Shape (chains, model_count + 1, components)
        # Calculate R-hat for the current cumulative set of models
        ess = diagnostics.effective_sample_size(cumulative_components)
        
        # Ensure r_hat can be correctly stored in cumulative_r_hats
        cumulative_ess.append(ess)  # Store the result in the preallocated array

    return cumulative_ess


@profile
def process_samples(directory, dataset, number_of_chains, params_structure, rng_key, save_directory):
    files=os.listdir(directory)
    full_file_paths =[os.path.join(directory, file) for file in files]
    total_samples=len(files)
    num_models=int(total_samples/number_of_chains)
    num_components=4
    numbers = re.findall(r'\d+', files[0])  # \d+ matches one or more digits
     # Extract the last number, if any
    last_number = int(numbers[-1]) if numbers else None
    thinning=last_number
    #Batch size for log probability calculation
    batch_size=100
    components = np.empty((number_of_chains, num_models, num_components), dtype=np.float64)
    log_probabilities = np.empty((number_of_chains, num_models),dtype=np.float64)
    for i in range(number_of_chains):
        for j in range(num_models):
            index = j + i * num_models
            rng_key,batch_key= jax.random.split(rng_key, 2)
            batch=get_batch(dataset,batch_size,batch_key)
            try:
                network = eqx.tree_deserialise_leaves(full_file_paths[index], params_structure)
            except:
                print("Error at sample:",index)
            log_prob=logprob_fn(network,batch)
            log_probabilities[i,j]=log_prob
            components[i, j, :] = extract_single_model_components(network)
            del network
    
    if not number_of_chains==1:
        r_hats=cumulative_r_hats(components)
        plot_r_hats(r_hats,thinning,number_of_chains)
    ess=cumulative_ess(components)
    components_samples=[components[:,:,i].flatten() for i in range(num_components)]
    plot_components(components_samples,total_samples,thinning, save_directory)
    plot_correlations(components_samples,total_samples,thinning, save_directory)
    plot_ess(ess,thinning,number_of_chains, save_directory)
    plot_log_probabilities(log_probabilities,thinning,number_of_chains, save_directory)
    del components_samples
    del ess
    del log_probabilities
    if 'r_hats' in locals():
        del r_hats
    gc.collect()

#Log probabilities graphs for samples coming from different SGLD variants 
def compare_methods(directories, dataset, params_structure, rng_key, save_directory):
    all_log_probabilities = []
    thinning=0
    for directory in directories:
        # List files in the current directory
        files = os.listdir(directory)
        full_file_paths = [os.path.join(directory, file) for file in files]
        total_samples=len(files)
        numbers = re.findall(r'\d+', files[0])  # \d+ matches one or more digits
        batch_size=100

        # Extract the last number from the first file to determine thinning
        numbers = re.findall(r'\d+', files[0])  # \d+ matches one or more digits
        last_number = numbers[-1] if numbers else None
        thinning = int(last_number)
        
        # Initialize arrays for log probabilities and components
        log_probabilities = np.empty((total_samples), dtype=np.float64)
        for index in range(total_samples):
        
            rng_key, batch_key = jax.random.split(rng_key, 2)
            batch = get_batch(dataset, batch_size, batch_key)
            
            # Deserialize the neural network from the file
            network = eqx.tree_deserialise_leaves(full_file_paths[index], params_structure)
            
            # Compute log probability
            log_prob = logprob_fn(network, batch)
            log_probabilities[index] = log_prob
            
            # Cleanup
            del network
        
        # Store results for this directory
        all_log_probabilities.append(log_probabilities)
        
    plot_log_prob_different_models(all_log_probabilities, thinning, save_directory)
    
    
    
    


dataset=CombinedTimeStepDataset(data_spatial_points, data_mag_values, data_phase_values, data_time_values)
if __name__ == "__main__":
    directories=("test","models")
    compare_methods(directories, dataset, params_init, rng_key, "test")
    
#Idea to make it faster. Change the plot accordingly. Try yielding as well
"""if model_count % thinning_interval == 0:
    cumulative_components = components[:, :model_count + 2, :]
    r_hat = diagnostics.potential_scale_reduction(cumulative_components)
    cumulative_r_hats.append(r_hat)""" 