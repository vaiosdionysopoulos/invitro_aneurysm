"""The first SGLD implementation for our model. Employing las.scan, it calculates sgld samples 
over the indicated iteartions and then discards samples according to our choice of burn-in period
and thinning factor"""
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
import psutil
import functools as ft
from functools import partial
import sys
import equinox as eqx
import plots_MCMC
from plots_MCMC import plot_correlations,plot_components
import sys
rng_key=jax.random.PRNGKey(0)

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

num_classes=2
rng_key, sub_key_vx, sub_key_vy, sub_key_vz, sub_key_geom, sub_key_lmda = jax.random.split(rng_key, 6)
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
    "mu": jnp.array((1.1, 0.1)),
    "sigma": sigma_mag * jnp.ones(num_classes),
    "sigmas_v": jnp.array([[sigma_phase_x, sigma_phase_y, sigma_phase_z]]) * jnp.ones((num_classes, 3)),
}

def lr_schedule(iter_, a0=0.05, a1=0.51, c=0.1, warmup_iters=100):
    lr = jax.lax.cond(
        iter_ < warmup_iters,
        lambda x: a0 * (x / warmup_iters),  # Warmup: linear increase
        lambda x: a0 / (x + c) ** a1,        # Decay schedule
        iter_  # The value to pass to the lambda functions
    )
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
    print("g_prob_log:",g_prob_log)
    # Evaluate velocity field at x
    vel_v_x = nn_vel_v_x(x)
    vel_v_y = nn_vel_v_y(x)
    vel_v_z = nn_vel_v_z(x)
    velocity = jnp.concatenate((vel_v_x, vel_v_y, vel_v_z))
    print("Velocity:",velocity)
    print("All good")
    # Magnitude data loss calculation
    log_normalizer_mag = -0.5 * jnp.log(2 * jnp.pi * sigma**2)
    log_exp_terms_mag = -0.5 * ((y_mag - mu) ** 2) / sigma**2
    log_terms_mag = log_normalizer_mag + log_exp_terms_mag + g_prob_log
    loss_mag = -jax.scipy.special.logsumexp(log_terms_mag)
    print("Loss mag:",loss_mag)
    # Phase (velocity) data loss calculation
    log_normalizer_phase = -0.5 * jnp.log(2 * jnp.pi * sigmas_v**2) # Shape (num_classes, 3)
    log_exp_terms_phase = -0.5 * ((y_phase - velocity) ** 2) / sigmas_v**2  # Shape (num_classes, 3)
    # Sum log terms over the velocity components
    temp = (log_normalizer_phase + log_exp_terms_phase).sum(axis=1)  # Shape (num_classes,)
    log_terms_phase = temp + g_prob_log
    loss_phase = -jax.scipy.special.logsumexp(log_terms_phase)
    print("Loss phase:",loss_phase)
    return loss_phase+loss_mag 

def l2_norm(params):  
    l2_norm = jtu.tree_reduce(
        lambda x, y: jnp.sum(x) + jnp.sum(y),
        jtu.tree_map(lambda x: x ** 2, params),
        0.0
    )
    return l2_norm
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

#Δεν το χω χρησιμοποιήσει ακόμα γιατί είναι πιο περίπλοκο και θα πρέπει να αλλάξω και το sgld.step
#΄ώστε να δέχεται iter_
def gradient_clip_threshold(iter_, max_clip=1, decay_factor=0.99):
    return max_clip * (decay_factor ** iter_)
@eqx.filter_jit
def grad_estimator(p, batch, clip_val=3.):
    """ Compute clipped estimate of gradient of log probability """
    # Call the JIT compiled gradient function
    _,g =eqx.filter_value_and_grad(logprob_fn)(p,batch)
    
    # Clipping the gradient
    return jtu.tree_map(ft.partial(jnp.clip, a_min=-clip_val, a_max=clip_val), g)

def samples_from_posterior_sgld(theta_init,
                           batch_size,
                           total_iter, thinning_factor,burn, lr_package,
                           rng_key,temperature=1.0):
    """
    SGLD sampler for Bayesian neural network parameters

    inputs:
    theta_init -> neural network parameter starting point
    static_model -> static part of eqx module

    batch_size -> how many data points to batch for the likelihood

    total_iter, thinning_factor, burn, lr_package -> SGLD sampler specifications

    returns -> pytree of SGLD samples of posterior
    """
    
    a1, a2, c = lr_package
    dataset=CombinedTimeStepDataset(data_spatial_points, data_mag_values, data_phase_values, data_time_values)

    # build the SGLD sampler now
    sgld = blackjax.sgld(grad_estimator)
    sgld_step = eqx.filter_jit(sgld.step)
    position = sgld.init(theta_init)
  
    @eqx.filter_jit
    def sgld_step_scan(carry, x):
        """One step of SGLD"""
        iter_, position, rng_key = carry
        rng_key, batch_key, sample_key = jax.random.split(rng_key, 3)

        data_batch=get_batch(dataset,batch_size,batch_key)
        
        lr = lr_schedule(iter_, a1, a2, c)
        return (iter_ + 1, sgld_step(sample_key, position, data_batch, lr,temperature), rng_key), position
     
     
    init = (0,position,rng_key)
    
    _, sgld_samples = lax.scan(sgld_step_scan, init, None, length = total_iter)

    return jtu.tree_map(lambda x: x[burn::thinning_factor], sgld_samples)


lr_package_SGLD = (0.01, 0.51, 0.001)
theta_init= eqx.filter(params_init, eqx.is_array)
batch_size_SGLD=int(0.05*num_obs)
total_iter_SGLD=50
thinning_factor_SGLD=5
burn_SGLD=0
if __name__=="__main__":
    def main():
        # Set up argument parsing
        
        samples=samples_from_posterior_sgld(theta_init,
                                                batch_size_SGLD,
                                                total_iter_SGLD, thinning_factor_SGLD, burn_SGLD, lr_package_SGLD,
                                                rng_key)
        
        eqx.tree_serialise_leaves("runs/samples.eqx", samples) 

    # Loading and deserializing the PyTrees from the file
        
        
        loaded_pytree_dict =eqx.tree_deserialise_leaves("/workspaces/codespaces-blank/invitro_aneurysm/runs/samples.eqx", samples) 
        geometry=loaded_pytree_dict["nn_geom"]
        velocity_x=loaded_pytree_dict["nn_vel_v_x"]
        velocity_y=loaded_pytree_dict["nn_vel_v_x"]
        velocity_z=loaded_pytree_dict["nn_vel_v_z"]
        print(type(geometry))
        def print_pytree_info(pytree):
            def print_info(leaf):
                if isinstance(leaf, jnp.ndarray):
                    print(f"Array with shape: {leaf.shape}, dtype: {leaf.dtype}")
                else:
                    print(f"Non-array value: {leaf}")
            
        # Apply the print_info function to each leaf in the pytree
            jax.tree_util.tree_map(print_info, pytree)
        print_pytree_info(geometry)

        leaves_geom,_=jax.tree_util.tree_flatten(geometry)
        leaves_vel_x,_=jax.tree_util.tree_flatten(velocity_x)
        leaves_vel_y,_=jax.tree_util.tree_flatten(velocity_y)
        leaves_vel_z,_=jax.tree_util.tree_flatten(velocity_z)
        random_leaf1=leaves_geom[2]
        random_leaf2=leaves_vel_x[6]
        random_leaf3=leaves_vel_y[14]
        random_leaf4=leaves_vel_z[7]
        component1=random_leaf1[:,5,5]
        component2=random_leaf2[:,15]
        component3=random_leaf3[:,89,16]
        component4=random_leaf4[:,53]
        
        plot_components(component1,component2,component3,component4,total_iter_SGLD)
        plot_correlations(component1,component2,component3,component4,total_iter_SGLD)

        
    main()