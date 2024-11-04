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

#I have to make the titles be dynamically defined depending on the statistic I want 

def plot_components(samples,total_samples,thinning,directory):
    samples1,samples2,samples3,samples4=samples
    iterations=np.arange(1, total_samples*thinning+1,thinning)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # Plot the first graph
    axs[0, 0].plot(iterations, samples1, color='blue')
    axs[0, 0].set_title(r'$\theta_1$ over iterations')
    axs[0, 0].set_xlabel('iterations')
    axs[0, 0].set_ylabel(r'$\theta_1$')
    axs[0, 0].grid()
    sns.despine(ax=axs[0, 0], trim=True)
    # Plot the second graph
    axs[0, 1].plot(iterations,samples2, color='red')
    axs[0, 1].set_title(r'$\theta_2$ over iterations')
    axs[0, 1].set_xlabel('iterations')
    axs[0, 1].set_ylabel(r'$\theta_2$')
    axs[0, 1].grid()
    sns.despine(ax=axs[0, 1], trim=True)

    # Plot the third graph
    axs[1, 0].plot(iterations, samples3, color='green')
    axs[1, 0].set_title(r'$\theta_3$ over iterations')
    axs[1, 0].set_xlabel(r'$\theta_3$')
    axs[1, 0].set_ylabel('iterations')
    axs[1, 0].grid()
    sns.despine(ax=axs[1,0], trim=True)

    # Plot the fourth graph
    axs[1, 1].plot(iterations,samples4, color='magenta')
    axs[1, 1].set_title(r'$\theta_1$ over iterations')
    axs[1, 1].set_xlabel('iterations')
    axs[1, 1].set_ylabel(r'$\theta_4$')
    axs[1, 1].grid()
    sns.despine(ax=axs[1,1], trim=True)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    save_directory = directory  # Saves in the 'runs' folder in the current directory
    save_filename = f"Components_plot_random_components_samples_{total_samples}_thinning_{thinning}.png"                # Name of the file

    # Combine the directory and filename
    full_save_path = os.path.join(save_directory, save_filename)
    # Save the figure
    plt.savefig(full_save_path, dpi=300)
    
    

def plot_correlations(samples,total_samples,thinning,directory):

    samples1,samples2,samples3,samples4=samples
    fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))

    # Plot ACF for data1
    plot_acf(samples1,lags=len(samples1)-1, ax=axs1[0, 0])
    axs1[0, 0].set_title('ACF of 'r'$\theta_1$')

    # Plot PACF for data1
    plot_acf(samples2,lags=len(samples2)-1, ax=axs1[0, 1])
    axs1[0, 1].set_title('ACF of 'r'$\theta_2$')

    # Plot ACF for data2
    plot_acf(samples3,lags=len(samples3)-1, ax=axs1[1, 0])
    axs1[1, 0].set_title('ACF of 'r'$\theta_3$')

    # Plot PACF for data2
    plot_acf(samples4,lags=len(samples4)-1, ax=axs1[1, 1])
    axs1[1, 1].set_title('ACF of 'r'$\theta_4$')

    plt.title("Autocorrelation for each component chain ")

    plt.tight_layout()
  
    # Save the figure
    save_directory = directory  # Saves in the 'runs' folder in the current directory
    save_filename1 = f"autocorrelations_of_thetas_samples_{total_samples}_thinning{thinning}.png"

    # Combine the directory and filename
    full_save_path = os.path.join(save_directory, save_filename1)
    # Save the figure
    plt.savefig(full_save_path, dpi=300)

#Takes a list of arrays of the r_hat of the components and plots each component's r_hat
def plot_r_hats(r_hat,thinning, number_of_chains, directory):
    r1 = [r[0] for r in r_hat]
    r2 = [r[1] for r in r_hat]
    r3 = [r[2] for r in r_hat]
    r4 = [r[3] for r in r_hat]
    xs=np.arange(1,len(r1)*thinning+1,thinning)
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Plotting multiple lines
    plt.plot(xs, r1, label=r'$\theta_1$', color='blue')  # Line 1
    plt.plot(xs, r2, label=r'$\theta_2$', color='red')   # Line 2
    plt.plot(xs, r3, label=r'$\theta_3$', color='green') # Line 3
    plt.plot(xs, r4, label=r'$\theta_4$', color='brown') # Line 4


    # Add titles and labels
    plt.title(r'Value of $\hat{R}$ over number of samples')
    plt.xlabel('Iterations')
    plt.ylabel(r'$\hat{R}$')

    # Add a grid for better readability
    plt.grid()
    sns.despine(trim=True)
    plt.legend(title='Thetas')
    plt.axhline(y=1.1, color='gray', linestyle='--', label='Threshold (1.1)')  # Add horizontal line

    # Adjust layout to prevent overlap
    plt.tight_layout()
    number_of_models=len(r_hat)
    save_directory = directory  # Saves in the 'runs' folder in the current directory
    save_filename = f"R_hat_plots_for_{number_of_chains}_chains_{number_of_models}_models_each_thinning_{thinning}.png"               

    # Combine the directory and filename
    full_save_path = os.path.join(save_directory, save_filename)
    # Save the figure
    plt.savefig(full_save_path, dpi=300)
    


def plot_ess(ess, thinning, number_of_chains, directory):
    
   # Prepare the ESS values for each parameter
    ess_values = list(zip(*ess))  # Transpose to separate each parameter's ESS values
    xs = np.arange(1, len(ess_values[0])*thinning + 1, thinning)  # Prepare x-axis values

    plt.figure(figsize=(10, 6))  # Set the figure size

    # Define an array of colors for chains
    color_array = ['blue', 'red', 'green', 'brown', 'orange', 'purple', 'cyan', 'magenta']  # Extend as needed

    # Check if there is only one chain
    if number_of_chains == 1:
        plt.plot(xs, ess_values[0], label=r'$\theta_1$', color=color_array[0])  # Plot single chain
    else:
        # Plotting lines for each parameter based on number of models
        for i in range(len(ess_values)):
            color = color_array[i % len(color_array)]  # Cycle through colors
            plt.plot(xs, ess_values[i], label=f'Chain {i + 1}', color=color)  # Dynamic label for each chain

        # Add a legend only if there are multiple chains
        plt.legend(title='Thetas')  # Title for the legend

    # Add titles and labels
    plt.title('Value of ESS over number of samples')
    plt.xlabel('Iterations')
    plt.ylabel('ESS')

    # Add a grid for better readability
    plt.grid()
    sns.despine(trim=True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    number_of_models = len(ess[0])  # Number of models based on ess
    save_directory = directory  # Saves in the 'runs' folder in the current directory
    save_filename = f"ESS_plots_for_{number_of_chains}_chains_{number_of_models}_models_each_thinning_{thinning}.png"

    full_save_path = os.path.join(save_directory, save_filename)

    # Save the figure
    plt.savefig(full_save_path, dpi=300)
    plt.close()  # Close the plot to free memory if plotting multiple times


def plot_log_probabilities(log_probs, thinning, number_of_chains, directory):
    # Prepare the log-probabilities values
    log_probs_values = log_probs  # Assuming log_probs is shaped (chains, models)
    xs = np.arange(1, (log_probs_values.shape[1])*thinning + 1, thinning)  # Prepare x-axis values

    plt.figure(figsize=(10, 6))  # Set the figure size

    # Define an array of colors for chains
    color_array = ['blue', 'red', 'green', 'brown', 'orange', 'purple', 'cyan', 'magenta']  # Extend as needed

    # Check if there is only one chain
    if number_of_chains == 1:
        # Flatten the array for a single chain
        flattened_log_probs = log_probs_values.flatten()  # Flatten to a 1D array
        plt.plot(xs, flattened_log_probs, label='Log-Probabilities', color=color_array[0])  # Plot single chain
    else:
        # Plotting lines for each chain
        for i in range(number_of_chains):
            color = color_array[i % len(color_array)]  # Cycle through colors
            plt.plot(xs, log_probs_values[i], label=f'Chain {i + 1}', color=color)  # Dynamic label for each chain

        # Add a legend
        plt.legend(title='Chains')  # Title for the legend

    # Add titles and labels
    plt.title('Log-Probabilities for different chains')
    plt.xlabel('Iterations')
    plt.ylabel('Log-Probability')

    # Add a grid for better readability
    plt.grid()
    sns.despine(trim=True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    number_of_models = log_probs_values.shape[1]  # Number of models based on log_probs
    save_directory = directory  # Saves in the 'runs' folder in the current directory
    save_filename = f"log_prob_plots_for_{number_of_chains}_chains_{number_of_models}_models_each_thinning_{thinning}.png"

    full_save_path = os.path.join(save_directory, save_filename)

    # Save the figure
    plt.savefig(full_save_path, dpi=300)
    plt.close()  # Close the plot to free memory if plotting multiple times

def plot_log_prob_different_models(log_probs, thinning, directory):
    label_names = ['SGLD', 'PSGLD', 'CSGLD']
    colors = ['blue', 'green', 'red']

    plt.figure(figsize=(10, 6))

    # Plot each array in the list
    for idx in range(len(log_probs)):
        log_prob = log_probs[idx]
        label = label_names[idx]
        # Create a range of samples adjusted for thinning
        samples = np.arange(0, len(log_prob) * thinning, thinning)  # Adjust for thinning
        plt.plot(samples, log_prob, color=colors[idx], label=label)

    plt.xlabel('Samples')
    plt.ylabel('Log Probability')
    plt.title('Log Probability across methods')
    plt.legend(title="Methods")  # Show the legend
    plt.grid(True)  # Add grid for better readability
    sns.despine(trim=True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure  
    save_directory = directory  # Saves in the 'runs' folder in the current directory
    save_filename = f"log_prob_plots_for_different_methods_thinning_{thinning}.png"

    full_save_path = os.path.join(save_directory, save_filename)

    # Save the figure
    plt.savefig(full_save_path, dpi=300)
    plt.close()  # Close the plot to free memory if plotting multiple times

def plot_errors(errors,thinning,directory):
    samples= np.arange(0, len(errors) * thinning, thinning)
    fig,ax=plt.subplots()
    ax.plot(samples,errors)
    plt.xlabel('Samples')
    plt.ylabel('MSE of x-velocity at timestep 1')
    plt.title('MSE of velocity x over iterations')
    plt.grid(True)  # Add grid for better readability
    sns.despine(trim=True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure  
    save_directory = directory  # Saves in the 'runs' folder in the current directory
    save_filename = f"MSE_for_velocity_x_timestep_1_thinning_{thinning}.png"

    full_save_path = os.path.join(save_directory, save_filename)

    # Save the figure
    plt.savefig(full_save_path, dpi=300)
    plt.close()  # Close the plot to free memory if plotting multiple times