# MuJoCo Warp (MJWarp)

MJWarp is a GPU-optimized version of the [MuJoCo](https://github.com/google-deepmind/mujoco) physics simulator, designed for NVIDIA hardware.

> [!NOTE]
> MJWarp is in Beta and under active development:
> * MJWarp developers will triage and respond to [bug report and feature requests](https://github.com/google-deepmind/mujoco_warp/issues).
> * MJWarp is mostly feature complete but requires performance optimization, documentation, and testing.
> * The intended audience during Beta are physics engine enthusiasts and learning framework integrators.
> * Machine learning / robotics researchers who just want to train policies should wait for the [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) or [Isaac](https://isaac-sim.github.io/IsaacLab/main/index.html)/[Newton](https://github.com/newton-physics/newton) integrations, which are coming soon.

MJWarp uses [NVIDIA Warp](https://github.com/NVIDIA/warp) to circumvent many of the [sharp bits](https://mujoco.readthedocs.io/en/stable/mjx.html#mjx-the-sharp-bits) in [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html#). MJWarp will be integrated into both [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and [Newton](https://github.com/newton-physics/newton).

MJWarp is maintained by [Google DeepMind](https://deepmind.google/) and [NVIDIA](https://www.nvidia.com/).

# Installing for development

```bash
git clone https://github.com/google-deepmind/mujoco_warp.git
cd mujoco_warp
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
```

During early development, MJWarp is on the bleeding edge - you should install Warp and MuJoCo nightly:

```bash
pip install warp-lang --pre --upgrade -f https://pypi.nvidia.com/warp-lang/
pip install mujoco --pre --upgrade -f https://py.mujoco.org/
```

Then install MJWarp in editable mode for local development:

```
pip install -e .[dev,cuda]
```

Now make sure everything is working:

```bash
pytest
```

Should print out something like `XX passed in XX.XXs` at the end!

If you plan to write Warp kernels for MJWarp, please use the `kernel_analyzer` vscode plugin located in `contrib/kernel_analyzer`.
Please see the `README.md` there for details on how to install it and use it.  The same kernel analyzer will be run on any PR
you open, so it's important to fix any issues it reports.

# Compatibility

The following features are implemented:

| Category           | Feature                                                                                                 |
| ------------------ | --------------------------------------------------------------------------------------------------------|
| Dynamics           | Forward, Inverse                                                                                        |
| Transmission       | All                                                                                                     |
| Actuator Dynamics  | All except `USER`                                                                                       |
| Actuator Gain      | All except `USER`                                                                                       |
| Actuator Bias      | All except `USER`                                                                                       |
| Geom               | All (`MESH`&harr;`SDF` collisions are not yet implemented)                                              |
| Constraint         | All                                                                                                     |
| Equality           | All except `FLEX`, `DISTANCE`                                                                           |
| Integrator         | All except `IMPLICIT`                                                                                   |
| Cone               | All                                                                                                     |
| Condim             | All                                                                                                     |
| Solver             | All except `PGS`, `noslip`                                                                              |
| Fluid Model        | `BOX` only                                                                                              |
| Tendon Wrap        | All                                                                                                     |
| Sensors            | All except `GEOMDIST`, `GEOMNORMAL`, `GEOMFROMTO`, `PLUGIN`, `USER`                                     |
| Flex               | `VERTCOLLIDE`, `ELASTICITY`                                                                             |
| Mass matrix format | Sparse and Dense                                                                                        |
| Jacobian format    | `DENSE` only (row-sparse, no islanding yet)                                                             |

[Differentiability via Warp](https://nvidia.github.io/warp/modules/differentiability.html#differentiability) is not currently
available.

# Benchmarking

Benchmark as follows:

```bash
mjwarp-testspeed --function=step --mjcf=test_data/humanoid/humanoid.xml --batch_size=8192
```

To get a full trace of the physics steps (e.g. timings of the subcomponents) run the following:

```bash
mjwarp-testspeed --function=step --mjcf=test_data/humanoid/humanoid.xml --batch_size=8192 --event_trace=True
```

