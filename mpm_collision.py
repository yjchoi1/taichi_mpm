import numpy as np
import os
import json
import taichi as ti
import random
import utils
from tqdm import tqdm
from engine.mpm_solver import MPMSolver

def run_collision(i):
    # inputs
    domain_size = 1.0
    ncubes_min, ncubes_max = 2, 3
    min_distance = 0.01
    cube_size_range=(0.3, 0.35)
    vel_max, vel_min = -3.0, 3.0
    sim_space = [[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]]
    cube_gen_space = [[0.21, 0.79], [0.21, 0.79], [0.21, 0.79]]
    # because of the memory issue in GNS, the following resolution recommended.
    # limit of # particles are hard-coded based on this resolution
    sim_resolution = (32, 32, 32)
    nparticel_per_vol = 262152
    nparticle_limits = 20000
    # visualization
    is_realtime_vis = True
    if is_realtime_vis:
        gui = ti.GUI('MPM3D', res=512, background_color=0x112F41)
    save_path = "./results/"
    # simulation
    nsteps = 350
    mpm_dt = 0.0025
    gravity = -9.81


    ti.init(arch=ti.cuda)
    mpm = MPMSolver(res=sim_resolution, size=domain_size)

    # gen cubes
    cubes = utils.generate_cubes(2, space_size=sim_space, cube_size_range=(0.2, 0.4))

    velocity_for_cubes = []
    nparticles = int(0)
    for cube in cubes:
        velocity = [random.uniform(vel_min, vel_max) for _ in range(len(sim_space))]
        velocity_for_cubes.append(velocity)
        mpm.add_cube(
            lower_corner=[cube[0], cube[1], cube[2]],
            cube_size=[cube[3], cube[4], cube[5]],
            material=MPMSolver.material_sand,
            velocity=velocity)
        nparticles_per_cube = (cube[3] * cube[4] * cube[5]) * nparticel_per_vol
        nparticles += nparticles_per_cube

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

        if is_realtime_vis:
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
                             boundaries=sim_space,
                             timestep_stride=3)

    # save particle group info.
    sim_data = {"cubes": cubes, "velocity_for_cubes": velocity_for_cubes, "nparticles": nparticles}
    with open(f"{save_path}/particle_info{i}.json", "w") as outfile:
        json.dump(sim_data, outfile)




if __name__ == "__main__":
    n_trajectories = 5
    for i in range(n_trajectories):
        data = run_collision(i)




    # # save
    # if frame == 99:
    #     writer = ti.tools.PLYWriter(num_vertices=particles['position'].shape[0])
    #     writer.add_vertex_pos(
    #         particles['position'][:, 0], particles['position'][:, 1], particles['position'][:, 2])
    #     # writer.export_frame_ascii(frame, series_prefix)