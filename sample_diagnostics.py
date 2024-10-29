import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import blackjax.diagnostics as diagnostics 
import equinox as eqx
import os
import re 
from plots_MCMC import *
from SGLD_prototype import params_init
from memory_profiler import profile
import gc
from SGLD_prototype import get_batch,loss_data,l2_norm
from SGLD_sampling import dataset, num_obs


rng_key=jax.random.PRNGKey(2)

def extract_single_model_components(model):
    comp1=model["nn_geom"].layers[1].layers[0].v[10,78]
    comp2=model["nn_vel_v_x"].layers[1].layers[1].bias[45]
    comp3=model["nn_vel_v_y"].layers[1].layers[1].bias[234]
    comp4=model["nn_vel_v_z"].layers[1].layers[2].v[200,15]
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
def process_samples(directory, number_of_chains,rng_key):
    files=os.listdir(directory)
    full_file_paths =[os.path.join(directory, file) for file in files]
    total_samples=len(files)
    num_models=int(total_samples/number_of_chains)
    num_components=4
    numbers = re.findall(r'\d+', files[0])  # \d+ matches one or more digits
     # Extract the last number, if any
    last_number = numbers[-1] if numbers else None
    thinning=last_number
    batch_size=100
    components = np.empty((number_of_chains, num_models, num_components), dtype=np.float64)
    log_probabilities=components = np.empty((number_of_chains, num_models),dtype=np.float64)
    for i in range(number_of_chains):
        for j in range(num_models):
            index = j + i * num_models
            rng_key,batch_key= jax.random.split(rng_key, 2)
            batch=get_batch(dataset,batch_size,batch_key)
            network = eqx.tree_deserialise_leaves(full_file_paths[index], params_init)
            log_prob=logprob_fn(network,batch)
            log_probabilities[i,j]=log_prob
            components[i, j, :] = extract_single_model_components(network)
            del network
    
    if not number_of_chains==1:
        r_hats=cumulative_r_hats(components)
        plot_r_hats(r_hats,thinning,number_of_chains)
    ess=cumulative_ess(components)
    components_samples=[components[:,:,i].flatten() for i in range(num_components)]
    plot_components(components_samples,total_samples,thinning)
    plot_correlations(components_samples,total_samples,thinning)
    plot_ess(ess,thinning,number_of_chains)
    plot_log_probabilities(log_probabilities,thinning,number_of_chains)
    del components_samples
    del ess
    del log_probabilities
    if 'r_hats' in locals():
        del r_hats
    gc.collect()

if __name__ == "__main__":
    process_samples("models",4,rng_key)
    process_samples("models",1,rng_key)

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