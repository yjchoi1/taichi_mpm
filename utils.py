import numpy as np
import random
from engine.mpm_solver import MPMSolver
from matplotlib import pyplot as plt
from matplotlib import animation



def add_random_cubes(mpm_solver, ncubes, cube_size_range, velocity_range, space):
    # To avoid overlap, we need to keep track of existing cubes
    existing_cubes = []

    for _ in range(ncubes):
        # Randomly determine the size of the cube
        length_x = random.uniform(cube_size_range[0], cube_size_range[1])
        length_y = random.uniform(cube_size_range[0], cube_size_range[1])
        length_z = random.uniform(cube_size_range[0], cube_size_range[1])

        while True:
            # Randomly determine the lower corner of the cube
            x = random.uniform(space[0][0], space[0][1] - length_x)
            y = random.uniform(space[1][0], space[1][1] - length_y)
            z = random.uniform(space[2][0], space[2][1] - length_z)

            # Define the current cube
            new_cube = (x, y, z, x + length_x, y + length_y, z + length_z)

            # Check if the new cube overlaps with any existing cube
            if any(overlap(new_cube, cube) for cube in existing_cubes):
                continue

            # If not, add it to the list and break the loop
            existing_cubes.append(new_cube)
            mpm_solver.add_cube(lower_corner=[x, y, z],
                                cube_size=[length_x, length_y, length_z],
                                material=MPMSolver.material_sand,
                                velocity=[random.uniform(
                                    velocity_range[0], velocity_range[1]
                                ) for _ in range(len(space))])
            break

def overlap(cube1, cube2):
    # A simple overlap check for two 3D rectangular prisms
    return not (cube1[3] <= cube2[0] or cube1[4] <= cube2[1] or cube1[5] <= cube2[2] or
                cube1[0] >= cube2[3] or cube1[1] >= cube2[4] or cube1[2] >= cube2[5])

def T(a):
    phi, theta = np.radians(32), np.radians(10)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return np.array([u, v]).swapaxes(0, 1) + 0.5

def animation_from_npz(path, npz_name, save_name, boundaries, timestep_stride=5):

    data = dict(np.load(f"{path}/{npz_name}.npz", allow_pickle=True))
    for i, (sim, info) in enumerate(data.items()):
        positions = info[0]

    # make animation
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    def animate(i):
        fig.clear()
        # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=xboundary, ylim=yboundary)
        ax = fig.add_subplot(projection='3d', autoscale_on=False)
        ax.set_xlim(boundaries[0][0], boundaries[0][1])
        ax.set_ylim(boundaries[1][0], boundaries[1][1])
        ax.set_zlim(boundaries[2][0], boundaries[2][1])
        ax.scatter(positions[i][:, 0], positions[i][:, 2], positions[i][:, 1], s=1)
        ax.view_init(elev=20., azim=i*0.5)
        ax.grid(True, which='both')

    # Creat animation
    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, len(positions), timestep_stride), interval=10)

    ani.save(f'{path}/{save_name}.gif', dpi=100, fps=30, writer='imagemagick')
    print(f"Animation saved to: {path}/{save_name}.gif")