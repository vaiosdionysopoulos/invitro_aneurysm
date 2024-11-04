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

rng_key=jax.random.PRNGKey(0)

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
    "mu": jnp.array([1.1, 0.1]),
    "sigma": sigma_mag * jnp.ones(num_classes),
    "sigmas_v": jnp.array([[sigma_phase_x, sigma_phase_y, sigma_phase_z]]) * jnp.ones((num_classes, 3)),
}

#Just changed it to see what happens
def lr_schedule(iter_, a0=0.1, a1=0.4, c=0.2, warmup_iters=100, min_lr=1e-6, max_lr=0.8, min_denom=1e-8):
    lr = jax.lax.cond(
        iter_ < warmup_iters,
        lambda x: a0 * (x / warmup_iters),
        lambda x: a0 / jnp.maximum(x + c, min_denom) ** a1,
        iter_
    )
    lr = jnp.clip(lr, min_lr, max_lr)
    return jnp.nan_to_num(lr, nan=0.0)

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
    return -num_obs * jnp.mean(loss_data(position,x,y_mag,y_phase), axis = 0)+ 0.5e3 * l2_total

@eqx.filter_jit
def grad_estimator(p, batch, clip_val=1.):
    """ Compute clipped estimate of gradient of log probability """
    # Call the JIT compiled gradient function
    _,g =eqx.filter_value_and_grad(logprob_fn)(p,batch)
    
    # Clipping the gradient
    return jtu.tree_map(ft.partial(jnp.clip, a_min=-clip_val, a_max=clip_val), g)

def compute_grad_mag(pytree):
    leaf_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), pytree)
    total_norm = jnp.sqrt(sum(jnp.square(leaf) for leaf in jax.tree_util.tree_flatten(leaf_norms)[0]))
    return total_norm