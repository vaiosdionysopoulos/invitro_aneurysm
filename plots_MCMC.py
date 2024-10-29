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

def plot_components(samples,total_samples,thinning):
    samples1,samples2,samples3,samples4=samples
    iterations=np.arange(1, total_samples+1,1)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # Plot the first graph
    axs[0, 0].plot(iterations, samples1, 'b-')
    axs[0, 0].set_title(r'$\theta_1$ over iterations')
    axs[0, 0].set_xlabel('iterations')
    axs[0, 0].set_ylabel(r'$\theta_1$')
    axs[0, 0].grid()
    sns.despine(ax=axs[0, 0], trim=True)
    # Plot the second graph
    axs[0, 1].plot(iterations,samples2, 'r-')
    axs[0, 1].set_title(r'$\theta_2$ over iterations')
    axs[0, 1].set_xlabel('iterations')
    axs[0, 1].set_ylabel(r'$\theta_2$')
    axs[0, 1].grid()
    sns.despine(ax=axs[0, 1], trim=True)

    # Plot the third graph
    axs[1, 0].plot(iterations, samples3, 'g-')
    axs[1, 0].set_title(r'$\theta_3$ over iterations')
    axs[1, 0].set_xlabel(r'$\theta_3$')
    axs[1, 0].set_ylabel('iterations')
    axs[1, 0].grid()
    sns.despine(ax=axs[1,0], trim=True)

    # Plot the fourth graph
    axs[1, 1].plot(iterations,samples4, 'm-')
    axs[1, 1].set_title(r'$\theta_1$ over iterations')
    axs[1, 1].set_xlabel('iterations')
    axs[1, 1].set_ylabel(r'$\theta_4$')
    axs[1, 1].grid()
    sns.despine(ax=axs[1,1], trim=True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    if not os.path.exists("runs"):
        os.makedirs("runs")
    # Save the figure
    save_directory = os.path.join(os.getcwd(), "runs")  # Saves in the 'runs' folder in the current directory
    save_filename = f"Components_plot_random_components_samples_{total_samples}_thinning_{thinning}.png"                # Name of the file

    # Combine the directory and filename
    full_save_path = os.path.join(save_directory, save_filename)
    # Save the figure
    plt.savefig(full_save_path, dpi=300)
    
    

def plot_correlations(samples,total_samples,thinning):

    samples1,samples2,samples3,samples4=samples
    iterations=np.arange(1, total_samples+1, 1)
    fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))

    # Plot ACF for data1
    plot_acf(samples1,lags=iterations-1, ax=axs1[0, 0])
    axs1[0, 0].set_title('ACF of 'r'$\theta_1$')

    # Plot PACF for data1
    plot_acf(samples2,lags=iterations-1, ax=axs1[0, 1])
    axs1[0, 1].set_title('ACF of 'r'$\theta_2$')

    # Plot ACF for data2
    plot_acf(samples3,lags=iterations-1, ax=axs1[1, 0])
    axs1[1, 0].set_title('ACF of 'r'$\theta_3$')

    # Plot PACF for data2
    plot_acf(samples4,lags=iterations-1, ax=axs1[1, 1])
    axs1[1, 1].set_title('ACF of 'r'$\theta_4$')

    plt.tight_layout()
    if not os.path.exists("runs"):
        os.makedirs("runs")
    # Save the figure
    save_directory = os.path.join(os.getcwd(), "runs")  # Saves in the 'runs' folder in the current directory
    save_filename1 = f"autocorrelations_of_thetas_samples_{total_samples}_thinning{thinning}.png"

    # Combine the directory and filename
    full_save_path = os.path.join(save_directory, save_filename1)
    # Save the figure
    plt.savefig(full_save_path, dpi=300)

#Takes a list of arrays of the r_hat of the components and plots each component's r_hat
def plot_r_hats(r_hat,thinning, number_of_chains):
    if not os.path.exists("runs"):
        os.makedirs("runs")
    r1 = [r[0] for r in r_hat]
    r2 = [r[1] for r in r_hat]
    r3 = [r[2] for r in r_hat]
    r4 = [r[3] for r in r_hat]
    xs=np.arange(1,len(r1)+1,1)
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Plotting multiple lines
    plt.plot(xs, r1, label=r'$\theta_1$', color='blue')  # Line 1
    plt.plot(xs, r2, label=r'$\theta_2$', color='red')   # Line 2
    plt.plot(xs, r3, label=r'$\theta_3$', color='green') # Line 3
    plt.plot(xs, r4, label=r'$\theta_4$', color='brown') # Line 4


    # Add titles and labels
    plt.title(r'Value of $\hat{R}$ over number of samples')
    plt.xlabel('samples')
    plt.ylabel(r'$\hat{R}$')

    # Add a grid for better readability
    plt.grid()
    sns.despine(trim=True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    if not os.path.exists("runs"):
        os.makedirs("runs")
    # Save the figure
    number_of_models=len(r_hat)
    save_directory = os.path.join(os.getcwd(), "runs")  # Saves in the 'runs' folder in the current directory
    save_filename = f"R_hat_plots_for_{number_of_chains}_chains_{number_of_models}_models_each_thinning_{thinning}.png"               

    # Combine the directory and filename
    full_save_path = os.path.join(save_directory, save_filename)
    # Save the figure
    plt.savefig(full_save_path, dpi=300)
    


def plot_ess(ess,thinning,number_of_chains):
    if not os.path.exists("runs"):
        os.makedirs("runs")
    e1 = [elem[0] for elem in ess]
    e2 = [elem[1] for elem in ess]
    e3 = [elem[2] for elem in ess]
    e4 = [elem[3] for elem in ess]
    xs=np.arange(1,len(e1)+1,1)
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Plotting multiple lines
    plt.plot(xs, e1, label=r'$\theta_1$', color='blue')  # Line 1
    plt.plot(xs, e2, label=r'$\theta_2$', color='red')   # Line 2
    plt.plot(xs, e3, label=r'$\theta_3$', color='green') # Line 3
    plt.plot(xs, e4, label=r'$\theta_4$', color='brown') # Line 4


    # Add titles and labels
    plt.title('Value of ESS over number of samples')
    plt.xlabel('samples')
    plt.ylabel('ESS')

    # Add a grid for better readability
    plt.grid()
    sns.despine(trim=True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    if not os.path.exists("runs"):
        os.makedirs("runs")
    # Save the figure
    number_of_models=len(ess[0])
    save_directory = os.path.join(os.getcwd(), "runs")  # Saves in the 'runs' folder in the current directory
    save_filename = f"ESS_plots_for_{number_of_chains}_chains_{number_of_models}_models_each_thinning_{thinning}.png"               

    # Combine the directory and filename
    full_save_path = os.path.join(save_directory, save_filename)
    # Save the figure
    plt.savefig(full_save_path, dpi=300)
    

   