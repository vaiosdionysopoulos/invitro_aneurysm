"""Functions needed for SGLD sampling and oredictive analysis"""
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

def load_data():
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
    return data_spatial_points,data_time_values,data_mag_values,data_phase_values,sigma_mag,sigma_phase_x,sigma_phase_y,sigma_phase_z,num_obs

data_spatial_points,data_time_values,data_mag_values,data_phase_values,sigma_mag,sigma_phase_x,sigma_phase_y,sigma_phase_z,num_obs=load_data()

class CombinedTimeStepDataset(Dataset):
    def __init__(self, spatial_points, mag_values, phase_values, time_steps):
        self.spatial_points = spatial_points
        self.mag_values = mag_values
        self.phase_values = phase_values
        self.time_steps = time_steps
        self.combined_data, self.y_mag, self.y_phase = self.combine_data()

    def combine_data(self):
        combined_data = []
        y_mag = []
        y_phase = []
        for t_idx, t in enumerate(self.time_steps):
            for point_idx, point in enumerate(self.spatial_points):
                combined_x_and_t = np.append(point, t)
                combined_data.append(combined_x_and_t)
                y_mag.append(self.mag_values[t_idx, point_idx])
                y_phase.append(self.phase_values[t_idx, point_idx])
        return jnp.array(combined_data), jnp.array(y_mag), jnp.array(y_phase)

    def __len__(self):
        return len(self.combined_data)
    
    def __getitem__(self, idx):
        x_b = self.combined_data[idx]
        y_mag_b = self.y_mag[idx]
        y_phase_b = self.y_phase[idx]
        return (x_b,y_mag_b,y_phase_b)
    
    def get_all_points_for_time_step(self, time_step_idx):
        # Check if the time_step_idx is valid
        if time_step_idx < 0 or time_step_idx >= len(self.time_steps):
            raise IndexError("Time step index out of bounds.")

        # Calculate start and end indices for the specified time step
        start_idx = time_step_idx * len(self.spatial_points)
        end_idx = start_idx + len(self.spatial_points)

        # Retrieve all combined data, magnitudes, and phases for this time step
        combined_data_at_time_step = self.combined_data[start_idx:end_idx]
        y_mag_at_time_step = self.y_mag[start_idx:end_idx]
        y_phase_at_time_step = self.y_phase[start_idx:end_idx]

        return combined_data_at_time_step, y_mag_at_time_step, y_phase_at_time_step

def get_batch(combined_dataset: CombinedTimeStepDataset, batch_size, key):
        # Shuffle and get random batch indices
        inputs=combined_dataset.combined_data
        mags=combined_dataset.y_mag
        phases=combined_dataset.y_phase
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=num_obs-1)  # Use num_data_points
        # Select a batch of data
        batch_inputs = inputs[indices]
        batch_mag_values = mags[indices]
        batch_phase_values = phases[indices] 
        return batch_inputs, batch_mag_values, batch_phase_values 

def nn_init(in_size, out_size, num_fourier_features, width_size, depth, activation, key):
    _, sub_key_fourier, sub_key_modifiedMLP = jax.random.split(key, 3)
    
    nn_four = FourierFeaturesLayer(in_size=in_size, 
                                   num_fourier_features=num_fourier_features, 
                                   key=sub_key_fourier)

    nn_modified = MLPModified(
        in_size=nn_four.out_size,
        out_size=out_size,
        width_size=width_size,
        depth=depth,
#         reparam_config=None,  # No RWF
        activation=activation,
        key=sub_key_modifiedMLP
    )
    return eqx.nn.Sequential([nn_four, nn_modified])

def model_init(rng_key):
    num_classes=2
    rng_key, sub_key_vx, sub_key_vy, sub_key_vz, sub_key_geom, sub_key_press = jax.random.split(rng_key, 6)
    params_init = {
        "nn_vel_v_x": nn_init(in_size=4, 
                            out_size=1, 
                            num_fourier_features=128, 
                            width_size=256, 
                            depth=4, 
                            activation=jax.nn.tanh, 
                            key=sub_key_vx),
        "nn_vel_v_y": nn_init(in_size=4, 
                            out_size=1, 
                            num_fourier_features=128, 
                            width_size=256, 
                            depth=4, 
                            activation=jax.nn.tanh, 
                            key=sub_key_vy),
        "nn_vel_v_z": nn_init(in_size=4, 
                            out_size=1, 
                            num_fourier_features=128, 
                            width_size=256, 
                            depth=4, 
                            activation=jax.nn.tanh, 
                            key=sub_key_vz),
        "nn_geom": nn_init(in_size=4, 
                            out_size=num_classes, 
                            num_fourier_features=128, 
                            width_size=256, 
                            depth=4, 
                            activation=jax.nn.tanh, 
                            key=sub_key_geom),
        "nn_press":nn_init(in_size=4, 
                            out_size=1, 
                            num_fourier_features=128, 
                            width_size=256, 
                            depth=4, 
                            activation=jax.nn.tanh, 
                            key=sub_key_press),
        "mu": jnp.array([1.1, 0.1]),
        "sigma": sigma_mag * jnp.ones(num_classes),
        "sigmas_v": jnp.array([[sigma_phase_x, sigma_phase_y, sigma_phase_z]]) * jnp.ones((num_classes, 3)),
    }
    return params_init

#Just changed it to see what happens
def lr_schedule(iter_, a0=0.1, a1=0.4, c=0.2, warmup_iters=100, min_lr=1e-6, max_lr=0.8, min_denom=1e-8):
    lr = jax.lax.cond(
        iter_ < warmup_iters,
        lambda x: a0 * (x / warmup_iters),
        lambda x: a0 / jnp.maximum(x + c, min_denom) ** a1,
        iter_
    )
    lr = jnp.clip(lr, min_lr, max_lr)
    return lr

@partial(vmap, in_axes=(None,0,0,0))
def loss_data(
    params: dict,
    x: jnp.ndarray,
    y_mag: jnp.ndarray,
    y_phase: jnp.ndarray) -> jnp.ndarray:
    # Extract parameters and neural network predictions
    nn_geom = params["nn_geom"]
    nn_vel_v_x = params["nn_vel_v_x"]
    nn_vel_v_y = params["nn_vel_v_y"]
    nn_vel_v_z = params["nn_vel_v_z"]
    mu = params["mu"] 
    sigma = jax.nn.softplus(params["sigma"]) 
    sigmas_v = jax.nn.softplus(params["sigmas_v"])

    num_classes = mu.shape[0]
    # Evaluate geometry field at x
    g_logits = nn_geom(x)
    g_prob_log = jax.nn.log_softmax(g_logits)
    # Evaluate velocity field at x
    vel_v_x = nn_vel_v_x(x)
    vel_v_y = nn_vel_v_y(x)
    vel_v_z = nn_vel_v_z(x)
    velocity = jnp.concatenate((vel_v_x, vel_v_y, vel_v_z))
    # Magnitude data loss calculation
    log_normalizer_mag = -0.5 * jnp.log(2 * jnp.pi * sigma**2)
    log_exp_terms_mag = -0.5 * ((y_mag - mu) ** 2) / sigma**2
    log_terms_mag = log_normalizer_mag + log_exp_terms_mag + g_prob_log
    loss_mag = -jax.scipy.special.logsumexp(log_terms_mag)
    # Phase (velocity) data loss calculation
    log_normalizer_phase = -0.5 * jnp.log(2 * jnp.pi * sigmas_v**2) # Shape (num_classes, 3)
    log_exp_terms_phase = -0.5 * ((y_phase - velocity) ** 2) / sigmas_v**2  # Shape (num_classes, 3)
    # Sum log terms over the velocity components
    temp = (log_normalizer_phase + log_exp_terms_phase).sum(axis=1)  # Shape (num_classes,)
    log_terms_phase = temp + g_prob_log
    loss_phase = -jax.scipy.special.logsumexp(log_terms_phase)
    return loss_phase+loss_mag 


def l2_norm(params):  
    l2_norm = jtu.tree_reduce(
        lambda x, y: jnp.sum(x) + jnp.sum(y),
        jtu.tree_map(lambda x: x ** 2, params),
        0.0
    )
    return l2_norm

#x is a vector (x,y,z,t)
@eqx.filter_jit
@partial(jax.vmap, in_axes=(0, None))
def divergence(x, params):
    velx = lambda x : params["nn_vel_v_x"](x)[0]
    vely = lambda x : params["nn_vel_v_y"](x)[0]
    velz = lambda x : params["nn_vel_v_z"](x)[0]

    # Take gradients of velocities
    grad_velx = jax.grad(velx, argnums=0)
    grad_vely = jax.grad(vely, argnums=0)
    grad_velz = jax.grad(velz, argnums=0)

    # Find divergence
    divergence = grad_velx(x)[0] + grad_vely(x)[1] + grad_velz(x)[2]
    
    return jnp.array(divergence)

#Takes a point and returns the residual of the Navier-Stokes equation
@eqx.filter_jit
@partial(jax.vmap, in_axes=(0, None))
def navier_stokes_residual(x,params, reynolds=4000):
    velx = lambda x : params["nn_vel_v_x"](x)[0]
    vely = lambda x : params["nn_vel_v_y"](x)[0]
    velz = lambda x : params["nn_vel_v_z"](x)[0]
    pressure= lambda x : params["nn_press"](x)[0]
    # Take gradients of velocities
    grad_velx = jax.grad(velx, argnums=0)
    grad_vely = jax.grad(vely, argnums=0)
    grad_velz = jax.grad(velz, argnums=0)
    grad_pressure=jax.grad(pressure, argnums=0)
    
    grad_grad_velx=jax.jacfwd(grad_velx, argnums=0)
    grad_grad_vely=jax.jacfwd(grad_vely, argnums=0)
    grad_grad_velz=jax.jacfwd(grad_velz, argnums=0)

    grad_time=jnp.array([grad_velx(x)[3],grad_vely(x)[3],grad_velz(x)[3]])
    convecive_term=jnp.array([velx(x)*grad_velx(x)[:3] + vely(x)*grad_vely(x)[:3] + velz(x)*grad_velz(x)[:3]])
    pressure_term=grad_pressure(x)[:3]
    laplacian_term=-(1/reynolds)*(jnp.diagonal(grad_grad_velx(x))[:3]+jnp.diagonal(grad_grad_vely(x))[:3]+jnp.diagonal(grad_grad_velz(x))[:3])
    return jnp.sum(jnp.square(grad_time + convecive_term + pressure_term + laplacian_term))


def reference_pressure(params,point):
    pressure= lambda x: params["nn_press"](x)[0]
    return pressure(point)
    


"""ASK ALEX ABOUT THE SIGNS IN THE LAST LINE OF LOGPROB_FN. Also ask about the prior,
how can we have both a prior that stems from L1 regularization and one from the physics """
#beta=1/|Ω|T (constant that appears in the probability distribution of velocity field v)
@eqx.filter_jit
def logprob_fn(position, batch, beta1=100, beta2=100):
    # blackjax calls the parameters which get updated 'position'
    # batch are objects that can be passed between iterations

    batch_data=batch[0]
    batch_physics=batch[1]
    reference_point=batch[2]

    x=batch_data[0]
    y_mag=batch_data[1]
    y_phase=batch_data[2]
    l2_total = (
    l2_norm(position["nn_geom"]) + 
    l2_norm(position["nn_vel_v_x"]) + 
    l2_norm(position["nn_vel_v_y"]) + 
    l2_norm(position["nn_vel_v_z"])
)
    div=divergence(batch_physics, position)
    navier_stokes_res=navier_stokes_residual(batch_physics, position)
    return -num_obs * jnp.mean(loss_data(position,x,y_mag,y_phase), axis = 0) - 0.5e3 * l2_total - beta1*jnp.mean(jnp.square(div)) - beta2*jnp.mean(navier_stokes_res)-reference_pressure(position, reference_point)

@eqx.filter_jit
def grad_estimator(p, batch, clip_val=0.4):
    """ Compute clipped estimate of gradient of log probability """
    # Call the JIT compiled gradient function
    _,g =eqx.filter_value_and_grad(logprob_fn)(p,batch)
    g = jtu.tree_map(partial(jnp.nan_to_num, nan=1e-8), g)
    # Clipping the gradient
    return jtu.tree_map(ft.partial(jnp.clip, a_min=-clip_val, a_max=clip_val), g)


def collocation_grid(grid_size, grid_key, a, b):
    """
    Uniformly samples points in a 4D domain so the integral in the physics-informed
    loss can be cast as an expectation following the low-discrepancy sampler.
    
    inputs:
    grid_size -> how many points to sample
    grid_key -> rng_key
    a -> tuple of minvals (a1, a2, a3, a4) for the 4 dimensions
    b -> tuple of maxvals (b1, b2, b3, b4) for the 4 dimensions
    """
    
    samples = []

    # Loop over each dimension
    for i in range(4):
        # Create the low-discrepancy samples for this dimension
        u0 = jax.random.uniform(grid_key, shape=(1,), minval=0, maxval=1)
        dim_samples = jnp.mod(u0 + (jnp.arange(grid_size) + 1) / grid_size, 1)
        
        # Scale the samples to the correct range for this dimension
        scaled_samples = a[i] + (b[i] - a[i]) * dim_samples
        
        # Append to the samples list
        samples.append(scaled_samples)

    # Stack the 4D samples into a single array
    return jnp.stack(samples, axis=-1)



#Helper function produces the structure to deserialise saved pytrees with 
def produce_structure(params_template, reps):
    def body_fn(carry,_):
        return carry,carry
    init=params_template
    _,structure=lax.scan(body_fn,init,jnp.arange(reps))
    return structure


#This is working for iteratively stacking models together. It is compatible with the way 
#we're vmaping over models below 
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


@eqx.filter_jit
def eval_single_model(x, params):
    geom = lambda x : params["nn_geom"](x)
    velx = lambda x : params["nn_vel_v_x"](x)[0]
    vely = lambda x : params["nn_vel_v_y"](x)[0]
    velz = lambda x : params["nn_vel_v_z"](x)[0]
    press = lambda x : params["nn_press"](x)[0]
    
    return jnp.array([geom(x)[0],geom(x)[1],velx(x),vely(x),velz(x),press(x)])



@eqx.filter_vmap(in_axes = (eqx.if_array(0), None))
def eval_ensemble(model,x):
    return eval_single_model(x,model)


posterior_predictive_samples = jax.jit(vmap(eval_ensemble, (None,0), 1))


#When applied to the combined tree it returns an array of the form (models,components)
def extract_models_components(models):
    comp1=models["nn_geom"].layers[1].layers[0].v[:,2,189]
    comp2=models["nn_vel_v_x"].layers[1].layers[1].bias[:,243]
    comp3=models["nn_vel_v_y"].layers[1].layers[1].bias[:,6]
    comp4=models["nn_vel_v_z"].layers[1].layers[2].v[:,53,1]
    return jnp.array([comp1,comp2,comp3,comp4]).T





