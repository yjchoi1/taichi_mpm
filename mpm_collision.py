import numpy as np
import json
import argparse
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
    ndim = len(sim_space)
    sim_resolution = inputs["sim_resolution"]
    # because of the memory issue in GNS, the following resolution recommended.
    # limit of # particles are hard-coded based on this resolution
    nparticel_per_vol = int(np.prod([sim_resolution[dim]/domain_size*2 for dim in range(ndim)]))
    nsteps = inputs["nsteps"]
    mpm_dt = inputs["mpm_dt"]
    gravity = inputs["gravity"]
    # visualization & simulation inputs
    is_realtime_vis = inputs["visualization"]["is_realtime_vis"]
    save_path = inputs["save_path"]
    ndim = len(sim_space)

    # init visualizer
    if is_realtime_vis:
        if ndim == 3:
            gui = ti.GUI('MPM3D', res=512, background_color=0x112F41)
        elif ndim == 2:
            gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)
        else:
            raise ValueError("`ndim` should be 2 or 3")

    # init MPM solver
    ti.init(arch=ti.cuda, device_memory_GB=3.4)
    mpm = MPMSolver(inputs=inputs, res=sim_resolution, size=domain_size)

    if inputs["gen_cube_from_data"]["generate"] == inputs["gen_cube_randomly"]["generate"]:
        raise NotImplemented(
            "Cube generation method should either be one of `gen_cube_from_data` or `gen_cube_randomly`")

    # Gen cubes from data
    if inputs["gen_cube_from_data"]["generate"]:
        sim_input = inputs["gen_cube_from_data"]["sim_inputs"][i]
        if len(inputs["gen_cube_from_data"]["sim_inputs"]) != \
                len(range(inputs["id_range"][0], inputs["id_range"][1])):
            raise NotImplemented(f"Length of `sim_inputs` should match the length of `id_range`")
        # Mass regarding soft mass
        cubes = sim_input["mass"]["cubes"]
        velocity_for_cubes = sim_input["mass"]["velocity_for_cubes"]
        # Mass regarding rigid obstacles
        if "obstacles" in sim_input:
            obstacles = sim_input["obstacles"]["cubes"]
        else:
            obstacles = None

    # Random cube generation
    elif inputs["gen_cube_randomly"]["generate"]:
        # Make cubes for mass regarding soft mass
        rand_gen_inputs = inputs["gen_cube_randomly"]["sim_inputs"]
        ncubes = rand_gen_inputs["mass"]["ncubes"]
        min_distance_between_cubes = rand_gen_inputs["mass"]["min_distance_between_cubes"]
        cube_size_range = rand_gen_inputs["mass"]["cube_size_range"]
        vel_range = rand_gen_inputs["mass"]["vel_range"]
        cube_gen_space = rand_gen_inputs["mass"]["cube_gen_space"]
        nparticle_limits = rand_gen_inputs["mass"]["nparticle_limits"]

        cubes = utils.generate_cubes(
            n=random.randint(ncubes[0], ncubes[1]),
            space_size=cube_gen_space,
            cube_size_range=cube_size_range,
            min_distance_between_cubes=min_distance_between_cubes,
            density=nparticel_per_vol,
            max_particles=nparticle_limits)
        velocity_for_cubes = []
        for _ in cubes:
            velocity = [random.uniform(vel_range[d][0], vel_range[d][1]) for d in range(ndim)]
            velocity_for_cubes.append(velocity)

        # Make cubes for mass regarding rigid obstacles
        if "obstacles" in rand_gen_inputs:
            rand_gen_inputs = inputs["gen_cube_randomly"]["sim_inputs"]
            ncubes = rand_gen_inputs["obstacles"]["ncubes"]
            min_distance_between_cubes = rand_gen_inputs["obstacles"]["min_distance_between_cubes"]
            cube_size_range = rand_gen_inputs["obstacles"]["cube_size_range"]
            cube_gen_space = rand_gen_inputs["obstacles"]["cube_gen_space"]
            nparticle_limits = rand_gen_inputs["obstacles"]["nparticle_limits"]

            obstacles = utils.generate_cubes(
                n=random.randint(ncubes[0], ncubes[1]),
                space_size=cube_gen_space,
                cube_size_range=cube_size_range,
                min_distance_between_cubes=min_distance_between_cubes,
                density=nparticel_per_vol,
                max_particles=nparticle_limits)
        else:
            obstacles = None
    else:
        raise ValueError

    # TODO: need to overlap check between `cubes` and `obstacles`.

    for idx, cube in enumerate(cubes):
        mpm.add_cube(
            lower_corner=[cube[0], cube[1], cube[2]] if ndim == 3 else [cube[0], cube[1]],
            cube_size=[cube[3], cube[4], cube[5]] if ndim == 3 else [cube[2], cube[3]],
            material=MPMSolver.material_sand,
            velocity=velocity_for_cubes[idx])

    # Make particle type array
    n_soil_particles = mpm.particle_info()["position"].shape[0]
    particle_type_soil = np.full(n_soil_particles, 6)

    if obstacles is not None:
        for idx, cube in enumerate(obstacles):
            mpm.add_cube(
                lower_corner=[cube[0], cube[1], cube[2]] if ndim == 3 else [cube[0], cube[1]],
                cube_size=[cube[3], cube[4], cube[5]] if ndim == 3 else [cube[2], cube[3]],
                material=MPMSolver.material_stationary,
                velocity=[0, 0, 0] if ndim == 3 else [0, 0])

        # Make particle type array
        n_entire_particles = mpm.particle_info()["position"].shape[0]
        particle_type_obstacle = np.full(n_entire_particles - n_soil_particles, 3)

    nparticles = len(mpm.particle_info()["position"])

    if ndim == 3:
        mpm.add_surface_collider(point=(sim_space[0][0], 0.0, 0.0), normal=(1.0, 0.0, 0.0), friction=inputs["wall_friction"])
        mpm.add_surface_collider(point=(sim_space[0][1], 0.0, 0.0), normal=(-1.0, 0.0, 0.0), friction=inputs["wall_friction"])
        mpm.add_surface_collider(point=(0.0, sim_space[1][0], 0.0), normal=(0.0, 1.0, 0.0), friction=inputs["wall_friction"])
        mpm.add_surface_collider(point=(0.0, sim_space[1][1], 0.0), normal=(0.0, -1.0, 0.0), friction=inputs["wall_friction"])
        mpm.add_surface_collider(point=(0.0, 0.0, sim_space[2][0]), normal=(0.0, 0.0, 1.0), friction=inputs["wall_friction"])
        mpm.add_surface_collider(point=(0.0, 0.0, sim_space[2][1]), normal=(0.0, 0.0, -1.0), friction=inputs["wall_friction"])
        mpm.set_gravity(gravity)
    else:
        mpm.add_surface_collider(point=(sim_space[0][0], 0.0), normal=(1.0, 0.0, ), friction=inputs["wall_friction"])
        mpm.add_surface_collider(point=(sim_space[0][1], 0.0), normal=(-1.0, 0.0), friction=inputs["wall_friction"])
        mpm.add_surface_collider(point=(0.0, sim_space[1][0]), normal=(0.0, 1.0), friction=inputs["wall_friction"])
        mpm.add_surface_collider(point=(0.0, sim_space[1][1]), normal=(0.0, -1.0), friction=inputs["wall_friction"])

    # run simulation
    print(f"Running simulation {i}/{inputs['id_range'][1]}...")
    positions = []
    for frame in tqdm(range(nsteps)):
        mpm.step(mpm_dt)
        colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00], dtype=np.uint32)
        particles = mpm.particle_info()
        positions.append(particles["position"])

        if is_realtime_vis:
            if ndim == 3:
                # simple camera transform
                screen_x = ((particles['position'][:, 0] + particles['position'][:, 2]) / 2 ** 0.5) - 0.2
                screen_y = (particles['position'][:, 1])
                # screen_z = (np_x[:, 2])
                screen_pos = np.stack([screen_x, screen_y], axis=-1)
                gui.circles(utils.T(particles['position']), radius=1.5, color=0x66ccff)
                gui.show()
            if ndim == 2:
                gui.circles(particles['position'],
                            radius=1.5,
                            color=colors[particles['material']])
    positions = np.stack(positions)


    # Change axis of positions (y & z), since the render in matplotlib uses the opposite axis order
    if ndim == 3 and follow_taichi_coord == False:
        positions = positions[:, :, [0, 2, 1]]

    # Save as npz
    # TODO (yc):
    #  Add a feature that only samples the particles on the perimeter of obstacle when saving npz
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trajectories = {}
    if obstacles is not None:
        particle_types = np.concatenate((particle_type_soil, particle_type_obstacle))
    else:
        particle_types = particle_type_soil
    trajectories[f"trajectory{i}"] = (
        positions,  # position sequence (timesteps, particles, dims)
        particle_types)  # particle type (particles, )
    np.savez_compressed(f"{save_path}/trajectory{i}", **trajectories)
    print(f"Trajectory {i} has {positions.shape[1]} particles")
    print(f"Output written to: {save_path}/trajectory{i}")

    # gen animation and save
    if inputs["visualization"]["is_save_animation"]:
        if i % inputs["visualization"]["skip"] == 0:
            utils.animation_from_npz(path=save_path,
                                     npz_name=f"trajectory{i}",
                                     save_name=f"trajectory{i}",
                                     boundaries=sim_space,
                                     timestep_stride=5,
                                     follow_taichi_coord=follow_taichi_coord)

    sim_data = {
        "sim_id": i,
        "cubes": cubes,
        "velocity_for_cubes": velocity_for_cubes,
        "obstacles": obstacles,
        "nparticles": int(nparticles)
    }
    with open(f"{save_path}/particle_info{i}.json", "w") as outfile:
        json.dump(sim_data, outfile, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default="inputs.json", type=str, help="Input json file name")
    args = parser.parse_args()

    # input
    input_filename = args.input_file
    follow_taichi_coord = True
    f = open(input_filename)
    inputs = json.load(f)
    f.close()

    # save input file being used.
    if not os.path.exists(inputs['save_path']):
        os.makedirs(inputs['save_path'])
    with open(f"{inputs['save_path']}/{input_filename}", "w") as input_file:
        json.dump(inputs, input_file, indent=4)

    for i in range(inputs["id_range"][0], inputs["id_range"][1]):
        data = run_collision(i, inputs)
