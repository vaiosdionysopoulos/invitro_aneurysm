import jax.numpy as jnp
import jax
import jax.random as random

def potential_energy(q):
    """Example potential energy function (negative log-posterior)"""
    return 0.5 * jnp.sum(q**2)  # Example: Gaussian potential

def kinetic_energy(p, M):
    """Kinetic energy function"""
    return 0.5 * jnp.dot(p, jnp.linalg.solve(M, p))

def hamiltonian(q, p, M):
    """Total Hamiltonian"""
    return potential_energy(q) + kinetic_energy(p, M)

def leapfrog(q, p, epsilon, M):
    """Leapfrog integration step"""
    p -= 0.5 * epsilon * jax.grad(potential_energy)(q)  # Update momentum
    q += epsilon * jnp.linalg.solve(M, p)               # Update position
    p -= 0.5 * epsilon * jax.grad(potential_energy)(q)  # Update momentum again
    return q, p

def hmc_step(q, M, epsilon, num_steps, rng_key):
    """Perform a single HMC step"""
    p = random.normal(rng_key, shape=q.shape)  # Sample initial momentum
    q_new, p_new = q, p

    for _ in range(num_steps):
        q_new, p_new = leapfrog(q_new, p_new, epsilon, M)

    # Metropolis acceptance
    current_H = hamiltonian(q, p)
    proposed_H = hamiltonian(q_new, p_new)
    accept_prob = jnp.exp(current_H - proposed_H)
    if random.uniform(rng_key) < accept_prob:
        return q_new  # Accept the new state
    return q  # Reject, return the old state

# Example usage
key = random.PRNGKey(0)
q_init = jnp.array([1.0, 2.0])  # Initial position
M = jnp.eye(len(q_init))         # Mass matrix (identity for simplicity)
epsilon = 0.1                     # Step size
num_steps = 10                    # Number of leapfrog steps

# Run a single HMC step
new_q = hmc_step(q_init, M, epsilon, num_steps, key)
print(new_q)