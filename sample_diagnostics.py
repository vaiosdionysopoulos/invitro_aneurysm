"""Executes diagnostics for MCMC samples of neural networks"""
import numpy as np
import jax
from jax import vmap
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
lower_range=jnp.array([x_range[0], y_range[0], z_range[0], t_range[0]])
upper_range=jnp.array([x_range[1], y_range[1], z_range[1], t_range[1]])

def convergenve_statistics (directory, number_of_chains, rng_key, save_directory, params_structure=params_init, a=lower_range, b=upper_range):
    files=os.listdir(directory)
    files.sort(key=lambda x: os.path.getctime(os.path.join('test', x)))
    full_file_paths =[os.path.join(directory, file) for file in files]
    numbers = re.findall(r'\d+', files[0])  # \d+ matches one or more digits
     # Extract the last number, if any
    last_number = int(numbers[-1]) if numbers else None
    thinning=last_number
    samples_per_file=int(numbers[1])-int(numbers[0])
    
    total_samples=int(len(files)*samples_per_file)
    num_components=4
    

    def generate_random_values(key, lower_bounds, upper_bounds):
        return lower_bounds + (upper_bounds - lower_bounds) * jax.random.uniform(key, shape=lower_bounds.shape)


    num_points=5
    random_values_list = []
    for i in range(num_points):
        rng_key, subkey = jax.random.split(rng_key)  # Split the key for each iteration
        random_values = generate_random_values(subkey, a,b)
        random_values_list.append(random_values)

    # Convert the list of random values to a JAX array for easier handling
    random_points = jnp.array(random_values_list)
    
   
    #Batch size for log probability calculation
    params_template=produce_structure(params_structure,samples_per_file)
    components = np.empty((total_samples, num_components), dtype=np.float64)
    velocities_and_pressure=np.empty((total_samples, num_points, 4))
    
    file_index=0

    for path in full_file_paths:
        samples=eqx.tree_deserialise_leaves(path,params_template)
        components[file_index * samples_per_file : (file_index + 1) * samples_per_file,:]=extract_models_components(samples)
        velocities_and_pressure[file_index * samples_per_file : (file_index + 1) * samples_per_file,:,:]=posterior_predictive_samples(samples, random_points)[:,:,2:]
        file_index+=1
        del samples
        gc.collect()
    
    #Form the chains to calculate R hat and ESS
    components=np.stack(np.array_split(components, number_of_chains, axis=0))
    
    #Plot components, ESS and R hat accross iterations
    #if not number_of_chains==1:
     #   plot_r_hats(components,save_directory,"find_title.png")

    trace_plots(components,save_directory,"components_plots.png")
    plot_correlations(components,save_directory,"components_correlation_plots.png")
    plot_r_hat(components, save_directory, "r_hat.png")
    plot_ess(components, save_directory,"components_ess.png")

    del components
    gc.collect()

    #Plot velocities and pressure accross iterations
    points = [f"Point {i}" for i in range(num_points)]
    velx=np.expand_dims(velocities_and_pressure[:,:,0], axis=0)
    print(velx.shape)
    vely=np.expand_dims(velocities_and_pressure[:,:,1], axis=0)
    velz=np.expand_dims(velocities_and_pressure[:,:,2], axis=0)
    pressure=np.expand_dims(velocities_and_pressure[:,:,3], axis=0)
    trace_plots(velx,save_directory,"velx.png", True, var_names=points)
    trace_plots(vely,save_directory,"vely.png", True, var_names=points)
    trace_plots(velz,save_directory,"velz.png", True, var_names=points)
    trace_plots(pressure,save_directory,"pressure.png", True, var_names=points)

   



if __name__ == "__main__":
    convergenve_statistics("test",2,rng_key,"SGLD_graphs")