# Granular Flow Simulation Using `taichi_mpm`
Simulating sand mass collision using [taichi mpm](https://github.com/taichi-dev/taichi_elements)

## Input
Using `input.json` file, granular mass can either be generated randomly in a specified domain,
or can be placed manually.

```shell
{
    "save_path": "./sand3d_collision/",
    "id_range": [  # the id of simulations to generate
        0,
        10
    ],
    "domain_size": 1.0,  # the largest domain length (same for all dimension)
    "friction_angle": 35,
    "wall_friction": 0.43,
    "elastic_modulus": 2000000.0,
    "poisson_ratio": 0.3,
    "rho": 1800,
    "sim_space": [  # lower and upper boundary for each dimension 
        [
            0.2,
            0.8
        ],
        [
            0.2,
            0.8
        ],
        [
            0.2,
            0.8
        ]
    ],
    "sim_resolution": [
        32,
        32,
        32
    ],
    "nsteps": 350,  # number of forward steps
    "mpm_dt": 0.0025,  # time between forward steps
    "gravity": -9.81,
    "gen_cube_randomly": {
        "generate": true,
        "ncubes": [1, 3],
        "min_distance_between_cubes": 0.01,
        "cube_size_range": [0.15, 0.30],
        "vel_range": [-2.5, 2.5],
        "cube_gen_space":  [[0.21, 0.79], [0.21, 0.79], [0.21, 0.79]],
        "nparticle_limits": 20000
    },
    "gen_cube_from_data": {
        "generate": true,
        "sim_inputs": [
            {
                "id": 0,  # id of simulation
                "mass": {
                    "cubes": [
                        [
                            0.2,  # x corner 
                            0.2,  # y corner
                            0.2,  # z corner
                            0.2,  # x length
                            0.3,  # y length
                            0.4   # z length                        
                    ],
                    "velocity_for_cubes": [
                        [
                            1.0,  # x vel
                            1.0,  # y vel
                            1.5   # z vel
                        ]
                    ]
                },
                "obstacles": null  # may repeat the as what is written in "mass" to add obstacles
            },
            {
              "id": 1  
                ...  # may repeat that is written in id 0 to add more simulations
            }
        ]
    },
    "visualization": {
        "is_realtime_vis": false,
        "is_save_animation": true,
        "skip": 1
    }
}
```

## Output
The output is saved and `.npz` file. The code also saves simple `.gif` animation for the simulation. 

## Run
```shell
python3 mpm_collision.py
```

## Simulation Example
![Sand collision example](example.gif)



