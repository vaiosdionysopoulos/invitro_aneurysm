"""Plots of statistics and of autocorrelation. It takes 1 list with 4 arrays of parameters or statistics 
of our samples and saves the plots """
import equinox as eqx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import jax 
import os
from statsmodels.graphics.tsaplots import plot_acf
import jax
import jax.numpy as jnp
import arviz as az
import xarray as xr


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,             # Default font size for general text
    "axes.titlesize": 10,         # Default size for titles
    "axes.labelsize": 14,         # Default size for axis labels
    "axes.labelweight": "normal",   # Bold axis labels only
    "legend.fontsize": 12,        # Legend font size
    "xtick.labelsize": 10,        # X-axis tick label size
    "ytick.labelsize": 10         # Y-axis tick label size
})


def trace_plots(trace, save_directory, save_filename, is_combined=False, var_names=None, figsize=(12, 6), start=0):
    """
    Create a custom multi-trace plot with integer x-axis and starting from 0.
    
    Parameters:
    trace (numpy.ndarray): The trace data in the format (chains, models, components).
    var_names (list, optional): List of variable names to plot. If None, all variables are plotted.
    figsize (tuple, optional): Figure size in inches.
    start (int, optional): Starting index for the x-axis. Default is 0.
    
    Returns:
    matplotlib.figure.Figure: The figure object containing the custom multi-trace plot.
    """
    if trace.ndim != 3:
        raise ValueError("Input trace data must have 3 dimensions (chains, models, components).")
    
    num_chains = trace.shape[0]
    num_models = trace.shape[1]
    num_vars = trace.shape[2]
    
    # Adjust the number of rows to display 2 subplots per row
    num_rows = (num_vars + 1) // 2

    if var_names is None:
        var_names = [f"Parameter {i+1}" for i in range(num_vars)]
    
    if not is_combined:
        fig, axes = plt.subplots(num_rows ,2, figsize=figsize, sharex=True)
        
        for i, var in enumerate(var_names):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
                
            # Get the trace for the variable and model
            var_trace = trace[:, :, i].flatten()
                
            # Calculate the number of samples
            num_samples = trace.shape[1]*trace.shape[0]
            
            # Create the x-axis with integers
            x = np.arange(start, start + num_samples)
            
            
            ax.plot(x, var_trace)
            
            ax.set_title(var)
            ax.set_xlabel("Samples")
            ax.set_ylabel("Value")
            ax.set_xlim(start, start + num_samples - 1)
    else:
        fig, ax = plt.subplots(figsize=figsize)
            # Calculate the number of samples
        num_samples = trace.shape[1]*trace.shape[0]
            # Create the x-axis with integers
        x = np.arange(start, start + num_samples)
        for i, var in enumerate(var_names):
        
            # Get the trace for the variable and model
            var_trace = trace[:, :, i].flatten()
            
            ax.plot(x, var_trace, label=var)
            
        ax.set_xlabel("Samples")
        ax.set_ylabel("Value")
        ax.set_xlim(start, start + num_samples - 1)
        plt.legend(title="Variables", loc="upper left", fontsize="small")
        
        
    plt.tight_layout()
    plt.suptitle("SGLD Trace Plots")
    plt.tight_layout()
    
    # Define the save path
    full_save_path = os.path.join(save_directory, save_filename)
    
    # Save the figure
    plt.savefig(full_save_path, dpi=300)
 


def plot_correlations(components, save_directory, save_filename, var_names=None, figsize=(12, 6), start=0):

    num_params = components.shape[2]
    num_rows = (num_params + 1) // 2
    fig, axes = plt.subplots(num_rows ,2, figsize=figsize, sharex=True)
    
    if var_names is None:
        var_names = [f"Parameter {i+1}" for i in range(num_params)]

    for i, var in enumerate(var_names):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        param_samples = components[:, :, i].ravel()
        plot_acf(param_samples, ax=ax, lags=10, title=f"Autocorrelation for Parameter {i+1}")


    full_save_path = os.path.join(save_directory, save_filename)

    # Save the figure
    plt.savefig(full_save_path, dpi=300)

#Takes a list of arrays of the r_hat of the components and plots each component's r_hat
def plot_r_hat(trace, save_directory, save_filename, var_names=None, figsize=(12, 6)):
    num_parameters = trace.shape[2]
    samples=trace.shape[1]    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    xs=np.arange(3,samples,1)
    # Initialize list to store R-hat values
    r_hat_values = [[] for _ in range(num_parameters)]
      # Create parameter names if not provided
    parameter_names = var_names if var_names is not None else [f"Parameter {i+1}" for i in range(num_parameters)]
    # Calculate R-hat for each parameter
    for i in range(num_parameters):
        for j in range(4,samples+1):
            trace_da = xr.DataArray(
                trace[:, :j, i],
                dims=["chain", "draw"],
                coords={
                    "chain": range(trace.shape[0]),
                    "draw": range(j)
                }
            )
            
            # Convert to InferenceData and calculate R-hat
            idata = az.convert_to_inference_data(trace_da)
            r_hat = az.rhat(idata)
            r_hat_values[i].append(float(r_hat.to_array().values[0]))
    
  
        
        # Plot all R-hat values on same axes
        ax.plot(xs, r_hat_values[i],  linestyle='-', linewidth=2, markersize=8, label=parameter_names[i])

    # Customize plot
    ax.set_title(r"$\hat{R}$ Values for All Parameters")
    ax.set_xlabel("Draws", labelpad=15)  # Increased labelpad to move the xlabel up
    ax.set_ylabel(r"$\hat{R}$")
    ax.grid(True)

    
    # Add threshold line
    ax.axhline(y=1.1, color='r', linestyle='--', label='Convergence Threshold')
    
    # Add legend
    ax.legend()

    plt.tight_layout()
  
    # Combine the directory and filename
    full_save_path = os.path.join(save_directory, save_filename)
    # Save the figure
    plt.savefig(full_save_path, dpi=300)
    

#Components should have the structure  (chains,models,components). If I have one chain, chain=1
def plot_ess(trace, save_directory, save_filename, var_names=None):
    chains, num_samples, variables = trace.shape
    # Initialize lists for storing cumulative ESS values per variable
    cumulative_bulk_ess = [[] for _ in range(variables)]
    cumulative_tail_ess = [[] for _ in range(variables)]

    # Create parameter names if not provided
    variable_names = var_names if var_names is not None else [f"Parameter {i+1}" for i in range(variables)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Iterate over each variable/parameter
    for i in range(variables):
        # Iterate over each sample
        for j in range(4, num_samples + 1):
            # Create DataArray with chains and draws, and explicitly set the variable name
            trace_da = xr.DataArray(
                trace[:, :j, i],
                dims=["chain", "draw"],
                coords={
                    "chain": range(trace.shape[0]),
                    "draw": range(j)
                }
            )
            
            # Convert to InferenceData and calculate R-hat
            idata = az.convert_to_inference_data(trace_da)


            # Calculate bulk and tail ESS
            bulk_ess = az.ess(idata)
            tail_ess = az.ess(idata, method="tail")

            # Store cumulative ESS values
            cumulative_bulk_ess[i].append(float(bulk_ess.to_array().values[0]))
            cumulative_tail_ess[i].append(float(tail_ess.to_array().values[0]))

        # Plot bulk ESS for this component
        ax1.plot(cumulative_bulk_ess[i], label=variable_names[i])

        # Plot tail ESS for this component
        ax2.plot(cumulative_tail_ess[i], label=variable_names[i])

    # Customize the plots
    ax1.set_title("Bulk Effective Sample Size (ESS)")
    ax1.set_xlabel("Draws")
    ax1.set_ylabel("Bulk ESS")
    ax1.axhline(y=num_samples, color='r', linestyle='--', label='Minimum Recommended ESS')
    ax1.legend(loc="upper left")

    ax2.set_title("Tail Effective Sample Size (ESS)")
    ax2.set_xlabel("Draws")
    ax2.set_ylabel("Tail ESS")
    ax2.axhline(y=num_samples, color='r', linestyle='--', label='Minimum Recommended ESS')
    ax2.legend(loc="upper left")

    plt.tight_layout()
    full_save_path = os.path.join(save_directory, save_filename)
    plt.savefig(full_save_path, dpi=300)
    plt.close()  # Close the plot to free memory if plotting multiple times


#r hats doesnt work, ess has problem with dimensions.
"""
# Create an ArviZ posterior plot
fig, ax = plt.subplots(figsize=(8, 6))
az.plot_posterior(idata, var_names=["param"], ax=ax)

# --- Customizations ---

# 1. Adjust Font Sizes
plt.title("Customized Posterior Plot", fontsize=16, fontweight='bold')
plt.xlabel("Parameter Value", fontsize=14)
plt.ylabel("Density", fontsize=14)

# 2. Add Vertical and Horizontal Lines
ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Line")
ax.axhline(0.1, color="blue", linestyle=":", linewidth=1.5, label="Horizontal Threshold")

# 3. Despine (remove top and right spines)
sns.despine(ax=ax, top=True, right=True)

# 4. Zoom Out by Adjusting x and y Limits
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
ax.set_xlim(x_min - 0.5, x_max + 0.5)  # Expanding the x-axis
ax.set_ylim(y_min - 0.05, y_max + 0.05)  # Expanding the y-axis

# 5. Adjust Legend
ax.legend(fontsize=12)

# 6. Use Matplotlib's tight_layout for better spacing
plt.tight_layout()

plt.show()



plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,             # Default font size for general text
    "axes.titlesize": 16,         # Default size for titles
    "axes.labelsize": 14,         # Default size for axis labels
    "axes.labelweight": "bold",   # Bold axis labels only
    "legend.fontsize": 12,        # Legend font size
    "xtick.labelsize": 10,        # X-axis tick label size
    "ytick.labelsize": 10         # Y-axis tick label size
})
"""