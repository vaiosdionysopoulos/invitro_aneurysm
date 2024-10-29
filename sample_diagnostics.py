import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import blackjax.diagnostics as diagnostics 
import equinox as eqx
import os
import re 
import plots_MCMC
from plots_MCMC import *
from SGLD_prototype import params_init

def break_chain(chain,number_of_chains):
    chains=[[] for _ in range(number_of_chains)]
    chain_length = len(chain)
    for i in range(number_of_chains):
        start_index = i * chain_length // number_of_chains
        end_index = (i + 1) * chain_length // number_of_chains
        chains[i] = chain[start_index:end_index]
    return chains

def extract_single_model_components(model):
    comp1=model["nn_geom"].layers[1].layers[0].v[10,78]
    comp2=model["nn_vel_v_x"].layers[1].layers[1].bias[45]
    comp3=model["nn_vel_v_y"].layers[1].layers[1].bias[234]
    comp4=model["nn_vel_v_z"].layers[1].layers[2].v[200,15]
    return jnp.array([comp1,comp2,comp3,comp4])

def extract_components(chains):
    num_chains = len(chains)
    num_models = len(chains[0]) if chains else 0
    num_components = len(extract_single_model_components(chains[0][0])) if num_models > 0 else 0

    # Preallocate array with shape (chains, models, components)
    components = np.empty((num_chains, num_models, num_components), dtype=np.float64)  # or another dtype if needed

    for i, chain in enumerate(chains):
        for j, model in enumerate(chain):
            components[i, j, :] = extract_single_model_components(model)

    return components

def process_samples(directory, number_of_chains):
    files=os.listdir(directory)
    total_samples=len(files)
    numbers = re.findall(r'\d+', files[0])  # \d+ matches one or more digits
     # Extract the last number, if any
    last_number = numbers[-1] if numbers else None
    thinning=last_number
    full_file_paths =[os.path.join(directory, file) for file in files]
    samples=[]
    for file in full_file_paths:
        network=eqx.tree_deserialise_leaves(file,params_init)
        samples.append(network)
    
    #Fix it from here and down so that the plot functions and the cumulative 
    #statistics ones match 
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


    chains=break_chain(samples,number_of_chains)
    components=extract_components(chains)
    r_hats=cumulative_r_hats(components)
    ess=cumulative_ess(components)
    components_samples=[components[:,:,i].flatten() for i in range(4)]
    plot_components(components_samples,total_samples,thinning)
    plot_correlations(components_samples,total_samples,thinning)
    plot_r_hats(r_hats,thinning,number_of_chains)
    plot_ess(ess,thinning,number_of_chains)

process_samples("models_no_thinning",4)

"""# Accessing parameters in FourierFeaturesLayer
B = geom.layers[0].B  # Accesses the B parameter in FourierFeaturesLayer

# Accessing parameters in MLPModified's u_layer
u_layer_weights = geom.layers[1].u_layer.v
u_layer_bias = geom.layers[1].u_layer.bias

# Accessing parameters in MLPModified's v_layer
v_layer_weights = geom.layers[1].v_layer.v
v_layer_bias = geom.layers[1].v_layer.bias

# Accessing weights and biases of individual layers in the layers list
layer1_weights = geom.layers[1].layers[0].v
layer1_bias = geom.layers[1].layers[0].bias

layer2_weights = geom.layers[1].layers[1].v
layer2_bias = geom.layers[1].layers[1].bias

layer3_weights = geom.layers[1].layers[2].v
layer3_bias = geom.layers[1].layers[2].bias

# Accessing the final layer's weights and biases
final_layer_weights = geom.layers[1].final_layer.v
final_layer_bias = geom.layers[1].final_layer.bias"""