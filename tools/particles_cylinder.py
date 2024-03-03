import numpy as np
import pandas as pd


def gen_particles_cylinder(center, radius, height, density, file_path):
    # Calculate the volume of the cylinder
    volume = np.pi * radius ** 2 * height

    # Calculate the total number of particles based on the given density
    num_particles = int(density * volume)

    # Empty array to store particle coordinates
    particles = np.zeros((num_particles, 3))

    # Generate uniformly distributed positions within the cylinder
    for i in range(num_particles):
        r_squared = np.random.uniform(0, radius ** 2)
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(0, height)

        # Convert to Cartesian coordinates and adjust for the center position
        x = np.sqrt(r_squared) * np.cos(theta) + center[0]
        y = np.sqrt(r_squared) * np.sin(theta) + center[1]
        z = z + center[2]  # Adjust if you want to change the base level

        particles[i] = [x, z, y]

    # Convert the particles array to a DataFrame and save as CSV
    df_particles = pd.DataFrame(particles, columns=['x', 'y', 'z'])
    df_particles.to_csv(file_path, index=False)

    return df_particles


gen_particles_cylinder([0.7, 0.7, 0.1], 0.2, 0.5, density=262144, file_path="../examples/cylinder_collapse/cylinder_particles.csv")



