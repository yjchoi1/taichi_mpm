import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.colors as colors


from typing import List, Tuple


import random
from typing import List, Tuple


def generate_random_cube(
        space_size=((0.2, 0.8), (0.2, 0.8), (0.2, 0.8)),
        cube_size_range=(0.1, 0.3)):

    cube_sizes = [random.uniform(*cube_size_range) for _ in range(3)]
    cube_starts = [random.uniform(space_size[i][0], space_size[i][1] - cube_sizes[i]) for i in range(3)]

    return (*cube_starts, *cube_sizes)

def check_overlap(cube1, cube2, min_distance_between_cubes=0.0):
    # Each cube is represented by (x_start, y_start, z_start, x_length, y_length, z_length)
    for i in range(3):
        if cube1[i] - min_distance_between_cubes >= cube2[i] + cube2[i+3] or cube1[i] + cube1[i+3] + min_distance_between_cubes <= cube2[i]:
            return False
    return True

def calculate_particles(cubes, density):
    total_volume = sum(c[3]*c[4]*c[5] for c in cubes)  # volume of each cube is length*width*height
    return total_volume * density

def generate_cubes(
        n,
        space_size=((0.2, 0.8), (0.2, 0.8), (0.2, 0.8)),
        cube_size_range=(0.1, 0.3),
        min_distance_between_cubes=0.01,
        density=2000,
        max_particles=float('inf')):
    cubes = []
    attemps = 0
    while len(cubes) < n:
        new_cube = generate_random_cube(space_size, cube_size_range)
        if any(check_overlap(
                new_cube, cube, min_distance_between_cubes=min_distance_between_cubes) for cube in cubes):
            cubes = []  # Clear the cube list and start over
        else:
            cubes.append(new_cube)
            if calculate_particles(cubes, density) > max_particles:
                return cubes[:-1]  # If adding the new cube results in exceeding the max particles, return the list without the last cube
        attemps += 1
        if attemps > 1000000:
            print(f"cannot find none-overlapping cubes in {attemps} attempts")
            break
    return cubes

def T(a):
    phi, theta = np.radians(32), np.radians(10)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return np.array([u, v]).swapaxes(0, 1) + 0.5

def animation_from_npz(path, npz_name, save_name, boundaries, timestep_stride=5, colorful=True):

    data = dict(np.load(f"{path}/{npz_name}.npz", allow_pickle=True))
    for i, (sim, info) in enumerate(data.items()):
        positions = info[0]

    # compute vel magnitude for color bar
    if colorful:
        initial_vel = np.zeros(positions[0].shape)
        initial_vel = initial_vel.reshape((1, initial_vel.shape[0], initial_vel.shape[1]))
        vel = positions[1:] - positions[:-1]
        vel = np.concatenate((initial_vel, vel))
        vel_magnitude = np.linalg.norm(vel, axis=-1)

    # make animation
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    def animate(i):
        print(f"Render step {i}/{len(positions)}")
        fig.clear()

        if colorful:
            cmap = plt.cm.viridis
            vmax = np.ndarray.flatten(vel_magnitude).max()
            vmin = np.ndarray.flatten(vel_magnitude).min()
            sampled_value =vel_magnitude[i]

        # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=xboundary, ylim=yboundary)
        ax = fig.add_subplot(projection='3d', autoscale_on=False)
        ax.set_xlim(boundaries[0][0], boundaries[0][1])
        ax.set_ylim(boundaries[1][0], boundaries[1][1])
        ax.set_zlim(boundaries[2][0], boundaries[2][1])
        if colorful:
            trj = ax.scatter(positions[i][:, 0], positions[i][:, 2], positions[i][:, 1],
                             c=sampled_value, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
            fig.colorbar(trj)
        else:
            ax.scatter(positions[i][:, 0], positions[i][:, 2], positions[i][:, 1],
                       s=1)
        ax.view_init(elev=20., azim=i*0.5)
        # ax.view_init(elev=20., azim=0.5)
        ax.grid(True, which='both')

    # Creat animation
    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, len(positions), timestep_stride), interval=20)

    ani.save(f'{path}/{save_name}.gif', dpi=100, fps=30, writer='imagemagick')
    print(f"Animation saved to: {path}/{save_name}.gif")