import numpy as np
import taichi as ti
import random
import utils
from tqdm import tqdm
from engine.mpm_solver import MPMSolver

inputs_for_sand = {
    "elastic_modulus": 1000000,
    "poisson_ratio": 0.45,
    "friction_angle": 30,
    "wall_friction": 0.2
}
sim_resolution = [128, 128]
domain_size = 1.0
nsteps = 1000
mpm_dt = 0.001
res = 512
ti.init(arch=ti.cuda, device_memory_GB=3.4)
write_to_disk = False
sim_space = [[0.1, 0.9], [0.1, 0.9]]

gui = ti.GUI("Taichi Elements", res=res, background_color=0x112F41)



ti.init(arch=ti.cuda, device_memory_GB=3.4)
mpm = MPMSolver(inputs=inputs_for_sand, res=sim_resolution, size=domain_size, unbounded=True)
mpm.add_surface_collider(point=(sim_space[0][0], 0.0), normal=(1.0, 0.0, ), friction=inputs_for_sand["wall_friction"])
mpm.add_surface_collider(point=(sim_space[0][1], 0.0), normal=(-1.0, 0.0), friction=inputs_for_sand["wall_friction"])
mpm.add_surface_collider(point=(0.0, sim_space[1][0]), normal=(0.0, 1.0), friction=inputs_for_sand["wall_friction"])
mpm.add_surface_collider(point=(0.0, sim_space[1][1]), normal=(0.0, -1.0), friction=inputs_for_sand["wall_friction"])

mpm.add_cube(
        lower_corner=[0.1, 0.5],
        cube_size=[0.3, 0.3],
        material=MPMSolver.material_elastic,
        velocity=[0, -1])


mpm.add_cube(
    lower_corner=[0.1, 0.1],
    cube_size=[0.8, 0.3],
    material=MPMSolver.material_water)

for frame in tqdm(range(nsteps)):
    mpm.step(mpm_dt)

    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00, 0xFFFF00],
                      dtype=np.uint32)
    particles = mpm.particle_info()
    gui.circles(particles['position'],
                radius=1.5,
                color=colors[particles['material']])
    gui.show(f'{frame:06d}.png' if write_to_disk else None)