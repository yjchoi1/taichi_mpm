import numpy as np
import os
import taichi as ti
import random
import utils
from tqdm import tqdm
from engine.mpm_solver import MPMSolver

def run_collision(i):
    # inputs
    domain_size = 1.0
    ncubes_min, ncubes_max = 2, 3
    cube_size_range=[0.25, 0.35]
    vel_max, vel_min = -3.5, 3.5
    sim_space = [[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]
    cube_gen_space = [[0.22, 0.78], [0.22, 0.78], [0.22, 0.78]]
    sim_resolution = (32, 32, 32)
    # visualization
    gui = ti.GUI('MPM3D', res=512, background_color=0x112F41)
    save_path = "./results/"
    # simulation
    nsteps = 380
    mpm_dt = 0.0025
    gravity = -9.81


    ti.init(arch=ti.cuda, device_memory_GB=4.0)
    mpm = MPMSolver(res=sim_resolution, size=domain_size)
    utils.add_random_cubes(mpm_solver=mpm,
                           ncubes=random.randint(ncubes_min, ncubes_max),
                           cube_size_range=cube_size_range,
                           velocity_range=[vel_min, vel_max],
                           space=cube_gen_space)
    mpm.add_surface_collider(point=(sim_space[0][0], 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    mpm.add_surface_collider(point=(sim_space[0][1], 0.0, 0.0), normal=(-1.0, 0.0, 0.0))
    mpm.add_surface_collider(point=(0.0, sim_space[1][0], 0.0), normal=(0.0, 1.0, 0.0))
    mpm.add_surface_collider(point=(0.0, sim_space[1][1], 0.0), normal=(0.0, -1.0, 0.0))
    mpm.add_surface_collider(point=(0.0, 0.0, sim_space[2][0]), normal=(0.0, 0.0, 1.0))
    mpm.add_surface_collider(point=(0.0, 0.0, sim_space[2][1]), normal=(0.0, 0.0, -1.0))
    mpm.set_gravity((0, gravity, 0))

    # run simulation
    print(f"Running simulation {i}/{n_trajectories}...")
    positions = []
    for frame in tqdm(range(nsteps)):
        mpm.step(mpm_dt)
        colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00], dtype=np.uint32)
        particles = mpm.particle_info()
        positions.append(particles["position"])

        # simple camera transform
        screen_x = ((particles['position'][:, 0] + particles['position'][:, 2]) / 2**0.5) - 0.2
        screen_y = (particles['position'][:, 1])
        # screen_z = (np_x[:, 2])
        screen_pos = np.stack([screen_x, screen_y], axis=-1)
        gui.circles(utils.T(particles['position']), radius=1.5, color=0x66ccff)
        gui.show()
    positions = np.stack(positions)

    # save as npz
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trajectories = {}
    trajectories[f"trajectory{i}"] = (
        positions,  # position sequence (timesteps, particles, dims)
        np.full(positions.shape[1], 6, dtype=int))  # particle type (particles, )
    np.savez_compressed(f"{save_path}/trajectory{i}", **trajectories)
    print(f"Trajectory {i} has {positions.shape[1]} particles")
    print(f"Output written to: {save_path}/trajectory{i}")

    # gen animation and save
    utils.animation_from_npz(path=save_path,
                             npz_name=f"trajectory{i}",
                             save_name=f"trajectory{i}",
                             boundaries=sim_space)

    return positions


if __name__ == "__main__":
    n_trajectories = 3
    for i in range(n_trajectories):
        positions = run_collision(i)




    # # save
    # if frame == 99:
    #     writer = ti.tools.PLYWriter(num_vertices=particles['position'].shape[0])
    #     writer.add_vertex_pos(
    #         particles['position'][:, 0], particles['position'][:, 1], particles['position'][:, 2])
    #     # writer.export_frame_ascii(frame, series_prefix)