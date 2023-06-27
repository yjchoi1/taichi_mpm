import numpy as np
import json
import os
import json
import taichi as ti
import random
import utils
from tqdm import tqdm
from engine.mpm_solver import MPMSolver

def run_collision(i, inputs):

    # inputs about general simulation information
    domain_size = inputs["domain_size"]
    sim_space = inputs["sim_space"]
    sim_resolution = inputs["sim_resolution"]
    # because of the memory issue in GNS, the following resolution recommended.
    # limit of # particles are hard-coded based on this resolution
    nparticel_per_vol = inputs["nparticel_per_vol"]  # 64*64*64/m^3
    nsteps = inputs["nsteps"]
    mpm_dt = inputs["mpm_dt"]
    gravity = inputs["gravity"]
    # visualization & simulation inputs
    is_realtime_vis = inputs["is_realtime_vis"]
    save_path = inputs["save_path"]

    # init visualizer
    if is_realtime_vis:
        gui = ti.GUI('MPM3D', res=512, background_color=0x112F41)

    # init MPM solver
    ti.init(arch=ti.cuda)
    mpm = MPMSolver(res=sim_resolution, size=domain_size)

    if inputs["gen_cube_from_data"]["generate"] == inputs["gen_cube_randomly"]["generate"]:
        raise NotImplemented(
            "Cube generation method should either be one of `gen_cube_from_data` or `gen_cube_randomly`")

    # Gen cubes from data
    if inputs["gen_cube_from_data"]["generate"]:
        cube_data = inputs["gen_cube_from_data"]["cube_data"]  # list containing cubes for each sim id
        if len(cube_data) != len(range(inputs["id_range"][0], inputs["id_range"][1])):
            raise NotImplemented(f"Enough length of cube_data should be provided to match id_range")
        cubes = cube_data[i]["cubes"]
        velocity_for_cubes = cube_data[i]["velocity_for_cubes"]

    # Random cube generation
    if inputs["gen_cube_randomly"]["generate"]:
        rand_gen_inputs = inputs["gen_cube_randomly"]
        ncubes = rand_gen_inputs["ncubes"]
        min_distance_between_cubes = rand_gen_inputs["min_distance_between_cubes"]
        cube_size_range = rand_gen_inputs["cube_size_range"]
        vel_range = rand_gen_inputs["vel_range"]
        cube_gen_space = rand_gen_inputs["cube_gen_space"]
        nparticle_limits = rand_gen_inputs["nparticle_limits"]

        # make cubes
        cubes = utils.generate_cubes(
            n=random.randint(ncubes[0], ncubes[1]),
            space_size=cube_gen_space,
            cube_size_range=cube_size_range,
            min_distance_between_cubes=min_distance_between_cubes,
            density=nparticel_per_vol,
            max_particles=nparticle_limits)
        # assign velocity for each cube
        velocity_for_cubes = []
        for _ in cubes:
            velocity = [random.uniform(vel_range[0], vel_range[1]) for _ in range(len(sim_space))]
            velocity_for_cubes.append(velocity)

    nparticles = int(0)
    for idx, cube in enumerate(cubes):
        mpm.add_cube(
            lower_corner=[cube[0], cube[1], cube[2]],
            cube_size=[cube[3], cube[4], cube[5]],
            material=MPMSolver.material_sand,
            velocity=velocity_for_cubes[idx])
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
    print(f"Running simulation {i} between {inputs['id_range'][0]} from {inputs['id_range'][1]}...")
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
    if inputs["is_save_animation"]:
        utils.animation_from_npz(path=save_path,
                                 npz_name=f"trajectory{i}",
                                 save_name=f"trajectory{i}",
                                 boundaries=sim_space,
                                 timestep_stride=3)

    # save particle group info.
    sim_data = {
            "sim_id": i, "cubes": cubes, "velocity_for_cubes": velocity_for_cubes, "nparticles": int(nparticles)
        }
    with open(f"{save_path}/particle_info{i}.json", "w") as outfile:
        json.dump(sim_data, outfile, indent=4)


if __name__ == "__main__":

    # load input file
    f = open('inputs.json')
    inputs = json.load(f)
    f.close()

    for i in range(inputs["id_range"][0], inputs["id_range"][1]):
        data = run_collision(i, inputs)