import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import animation
import random
import pandas as pd
from typing import Union

import engine.mpm_solver


def generate_random_cube(
        space_size,
        cube_size_range,
):
    """
    Make a cube which is defined as,
    [x_start, y_start, z_start, z_len, y_len, z_len]

    space_size: a domain where cube can be generated e.g., ((0.2, 0.8), (0.2, 0.8))
    cube_size_range: a range that defines random cube size.
      It can be
        1) Size ranges are defined for all dims. (e.g., [[0.15, 0.3], [0.15, 0.3], [0.15, 0.3]]
        2) Or, if you want it to be squared shape, [0.3, 0.5] which is the range of squared-shaped cube
        that will be generated.

    """
    ndim = len(space_size)
    try:
        # if size ranges are defined for all dims
        cube_sizes = [random.uniform(min_max[0], min_max[1]) for min_max in cube_size_range]
    except:
        # if only one size range is defined (e.g., squared shape)
        size = random.uniform(cube_size_range[0], cube_size_range[1])
        cube_sizes = [size for _ in range(ndim)]
    cube_starts = [random.uniform(space_size[i][0], space_size[i][1] - cube_sizes[i]) for i in range(ndim)]
    return (*cube_starts, *cube_sizes)

def check_overlap(cube1, cube2, min_distance_between_cubes=0.0):
    ndim = int(len(cube1) / 2)
    for i in range(ndim):
        if cube1[i] - min_distance_between_cubes >= cube2[i] + cube2[i + ndim] or \
           cube1[i] + cube1[i + ndim] + min_distance_between_cubes <= cube2[i]:
            return False
    return True

def calculate_particles(cubes, density):
    ndim = len(cubes[0]) / 2
    if ndim == 3:
        total_volume = sum(c[3] * c[4] * c[5] for c in cubes)
    elif ndim == 2:
        total_volume = sum(c[2] * c[3] for c in cubes)
    else:
        raise ValueError("Only 2D and 3D dimensions are supported.")
    return total_volume * density

def generate_cubes(n,
                   space_size,
                   cube_size_range,
                   min_distance_between_cubes,
                   density,
                   max_particles=float('inf')):
    """
    Make none-overlapping n number of cubes which is defined as,
    [x_start, y_start, z_start, z_len, y_len, z_len]

    space_size: a domain where cube can be generated e.g., ((0.2, 0.8), (0.2, 0.8))
    cube_size_range: a range that defines random cube size.
      It can be
        1) Size ranges are defined for all dims. (e.g., [[0.15, 0.3], [0.15, 0.3], [0.15, 0.3]]
        2) Or, if you want it to be squared shape, [0.3, 0.5] which is the range of squared-shaped cube
        that will be generated.
    min_distance_between_cubes: separation distance between cubes
    density: n particle per volume (n-particles/m^3)
    max_particles: restrict the numer of particles that will be generated.
    """
    cubes = []
    attempts = 0
    while len(cubes) < n:
        new_cube = generate_random_cube(space_size, cube_size_range)
        if not any(check_overlap(new_cube, cube, min_distance_between_cubes=min_distance_between_cubes) for cube in cubes):
            cubes.append(new_cube)
            if calculate_particles(cubes, density) > max_particles:
                return cubes[:-1]
        attempts += 1
        if attempts > 10000000:
            print(f"Cannot find non-overlapping cubes in {attempts} attempts")
    if len(cubes) == 0:
        raise Exception(
            f"The list `cubes` is empty meaning that fail to find non-overlapping cubes in {attempts} attempts")
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

def animation_from_npz(
        path,
        npz_name,
        save_name,
        boundaries,
        timestep_stride=5,
        colorful=True,
        follow_taichi_coord=False):

    data = dict(np.load(f"{path}/{npz_name}.npz", allow_pickle=True))
    for i, (sim, info) in enumerate(data.items()):
        positions = info[0]
    ndim = positions.shape[-1]

    # compute vel magnitude for color bar
    if colorful:
        initial_vel = np.zeros(positions[0].shape)
        initial_vel = initial_vel.reshape((1, initial_vel.shape[0], initial_vel.shape[1]))
        vel = positions[1:] - positions[:-1]
        vel = np.concatenate((initial_vel, vel))
        vel_magnitude = np.linalg.norm(vel, axis=-1)

    if ndim == 2:
        # make animation
        fig, ax = plt.subplots()

        def animate(i):
            fig.clear()
            # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=xboundary, ylim=yboundary)
            ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
            ax.set_xlim(boundaries[0][0], boundaries[0][1])
            ax.set_ylim(boundaries[1][0], boundaries[1][1])
            ax.scatter(positions[i][:, 0], positions[i][:, 1], s=1)
            ax.grid(True, which='both')

    if ndim == 3:
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
                sampled_value = vel_magnitude[i]

            if follow_taichi_coord:
                # Note: z and y is interchanged to match taichi coordinate convention.
                ax = fig.add_subplot(projection='3d', autoscale_on=False)
                ax.set_xlim(boundaries[0][0], boundaries[0][1])
                ax.set_ylim(boundaries[2][0], boundaries[2][1])
                ax.set_zlim(boundaries[1][0], boundaries[1][1])
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.set_zlabel("y")
                ax.invert_zaxis()
                if colorful:
                    trj = ax.scatter(positions[i][:, 0], positions[i][:, 2], positions[i][:, 1],
                                     c=sampled_value, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
                    fig.colorbar(trj)
                else:
                    ax.scatter(positions[i][:, 0], positions[i][:, 2], positions[i][:, 1],
                               s=1)
                ax.set_box_aspect(
                    aspect=(float(boundaries[0][0]) - float(boundaries[0][1]),
                            float(boundaries[2][0]) - float(boundaries[2][1]),
                            float(boundaries[1][0]) - float(boundaries[1][1])))
                ax.view_init(elev=20., azim=i*0.5)
                # ax.view_init(elev=20., azim=0.5)
                ax.grid(True, which='both')
            else:
                # Note: boundaries should still be permuted
                ax = fig.add_subplot(projection='3d', autoscale_on=False)
                ax.set_xlim(boundaries[0][0], boundaries[0][1])
                ax.set_ylim(boundaries[1][0], boundaries[1][1])
                ax.set_zlim(boundaries[2][0], boundaries[2][1])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.invert_zaxis()
                if colorful:
                    trj = ax.scatter(positions[i][:, 0], positions[i][:, 1], positions[i][:, 2],
                                     c=sampled_value, vmin=vmin, vmax=vmax, cmap=cmap, s=1)
                    fig.colorbar(trj)
                else:
                    ax.scatter(positions[i][:, 0], positions[i][:, 1], positions[i][:, 2],
                               s=1)
                ax.set_box_aspect(
                    aspect=(float(boundaries[0][0]) - float(boundaries[0][1]),
                            float(boundaries[1][0]) - float(boundaries[1][1]),
                            float(boundaries[2][0]) - float(boundaries[2][1])))
                ax.view_init(elev=20., azim=i * 0.5)
                # ax.view_init(elev=20., azim=0.5)
                ax.grid(True, which='both')

    # Creat animation
    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, len(positions), timestep_stride), interval=20)

    ani.save(f'{path}/{save_name}.gif', dpi=100, fps=30, writer='imagemagick')
    print(f"Animation saved to: {path}/{save_name}.gif")

def add_material_points(
        mpm_solver: engine.mpm_solver.MPMSolver,
        ndim,
        particles_to_add: Union[str, list],
        material: int,
        velocity: list
):
    # generate cube-shaped particles from the list defining the cube shape,
    #   e.g., for 2d case, cube = [x_min, y_min, len_x, len_y]
    if type(particles_to_add) is list or type(particles_to_add) is tuple:
        mpm_solver.add_cube(
            lower_corner=[
                particles_to_add[0], particles_to_add[1], particles_to_add[2]] if ndim == 3 else [particles_to_add[0], particles_to_add[1]],
            cube_size=[
                particles_to_add[3], particles_to_add[4], particles_to_add[5]] if ndim == 3 else [particles_to_add[2], particles_to_add[3]],
            material=material,
            velocity=velocity)
    # generate particles from user defined csv files containing particle coordinate
    elif type(particles_to_add) is str:
        particle_coords = read_particles(particles_to_add)
        if particle_coords.shape[-1] != ndim:
            raise ValueError(
                f"Particle file is {particle_coords.shape[-1]}d data, but sim space is {ndim}d")
        mpm_solver.add_particles(
            particles=particle_coords,
            material=material,
            velocity=velocity
        )
    else:
        raise ValueError("Wrong input type for particle gen")


def read_particles(path):
    df = pd.read_csv(path, header=1)
    return df.to_numpy()
