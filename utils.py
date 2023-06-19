import numpy as np
import random
from engine.mpm_solver import MPMSolver
from matplotlib import pyplot as plt
from matplotlib import animation


def add_random_cubes(mpm_solver, ncubes, min_distance, cube_size_range, velocity_range, space, nparticel_per_vol, min_nparticles,
                     max_nparticles, max_attempts):
    # To avoid overlap, we need to keep track of existing cubes
    existing_cubes = []
    vel_for_cubes = []
    total_nparticles = 0
    for _ in range(ncubes):
        attempts = 0
        while attempts < max_attempts:
            # Randomly determine the size of the cube
            length_x = random.uniform(cube_size_range[0], cube_size_range[1])
            length_y = random.uniform(cube_size_range[0], cube_size_range[1])
            length_z = random.uniform(cube_size_range[0], cube_size_range[1])

            # compute approximate # particles
            nparticles = nparticel_per_vol * (length_x * length_y * length_z)

            # Randomly determine the lower corner of the cube
            start_x = random.uniform(space[0][0], space[0][1] - length_x)
            start_y = random.uniform(space[1][0], space[1][1] - length_y)
            start_z = random.uniform(space[2][0], space[2][1] - length_z)

            # Define the current cube
            new_cube = (start_x, start_y, start_z,
                        start_x + length_x, start_y + length_y, start_z + length_z)

            # Check if the new cube overlaps with any existing cube
            if any(overlap(new_cube, cube, min_distance=min_distance) for cube in existing_cubes) or max_nparticles < total_nparticles + nparticles:
                existing_cubes = []
                total_nparticles = 0
                attempts += 1
                continue

            # If not, add it to the list and break the loop
            existing_cubes.append(new_cube)
            total_nparticles += nparticles
            vel = [random.uniform(velocity_range[0], velocity_range[1]) for _ in range(len(space))]
            vel_for_cubes.append(vel)
            mpm_solver.add_cube(lower_corner=[start_x, start_y, start_z],
                                cube_size=[length_x, length_y, length_z],
                                material=MPMSolver.material_sand,
                                velocity=vel)
            break

    data = {"ncubes": ncubes,
            "coord_for_cubes": existing_cubes,
            "vel_for_cubes": vel_for_cubes,
            "total_nparticles": total_nparticles}
    return data


def overlap(cube1, cube2, min_distance):
    """
    Checks if cube1 and cube2 are overlapping, with at least distance d between them.

    Each cube is defined as a tuple like (start_x, start_y, start_z, end_x, end_y, end_z).
    """
    # For each dimension
    for i in range(3):
        # If cube1's ending coordinate in this dimension (plus the distance d)
        # is less than or equal to cube2's starting coordinate in this dimension
        # OR cube1's starting coordinate in this dimension (minus the distance d)
        # is greater than or equal to cube2's ending coordinate in this dimension,
        # then the cubes are not overlapping in this dimension.
        if cube1[i+3] + min_distance <= cube2[i] or cube1[i] - min_distance >= cube2[i+3]:
            return False

    # If the cubes are overlapping in all dimensions
    return True

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
        fig, animate, frames=np.arange(0, len(positions), timestep_stride), interval=20)

    ani.save(f'{path}/{save_name}.gif', dpi=100, fps=30, writer='imagemagick')
    print(f"Animation saved to: {path}/{save_name}.gif")