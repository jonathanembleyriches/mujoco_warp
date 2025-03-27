# Copyright 2025 The Newton Developers
# Licensed under the Apache License, Version 2.0
import numpy as np
import sys

print("_multiarray_umath loaded:", "numpy._core._multiarray_umath" in sys.modules)
print("_umath_linalg loaded:", "numpy.linalg._umath_linalg" in sys.modules)
import logging
import time
from typing import Sequence

import mujoco
import numpy as np
import warp as wp
from absl import app
from absl import flags

import mujoco_warp as mjwarp
# wp.config.verify_cuda = True  #

# wp.config.verify_fp = True
_MODEL_PATH = flags.DEFINE_string(
    "mjcf", None, "Path to a MuJoCo MJCF file.", required=True
)
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool(
    "clear_kernel_cache", False, "Clear kernel cache (to calculate full JIT time)"
)
_ENGINE = flags.DEFINE_enum("engine", "mjwarp", ["mjwarp", "mjc"], "Simulation engine")
_LS_PARALLEL = flags.DEFINE_bool(
    "ls_parallel", False, "Engine solver with parallel linesearch"
)
@wp.func
def lookup(foos: wp.array(dtype=wp.vec3f), index: int):
    return foos[index]
@wp.kernel
def loss_kernel(xpos: wp.array2d(dtype=wp.vec3f),
                loss: wp.array(dtype=wp.float32)):
    # We use only one thread (tid==0) for this simple loss computation.
    if wp.tid() > 0:
        return

    diff = lookup(xpos, 0)[2]

    # Use an atomic add to accumulate the loss
    wp.atomic_add(loss, 0, diff)

def _main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print(f"Loading model from: {_MODEL_PATH.value}")
    if _MODEL_PATH.value.endswith(".mjb"):
        mjm = mujoco.MjModel.from_binary_path(_MODEL_PATH.value)
    else:
        mjm = mujoco.MjModel.from_xml_path(_MODEL_PATH.value)
    mjd = mujoco.MjData(mjm)

    mujoco.mj_forward(mjm, mjd)

    print("Engine: MuJoCo Warp")
    m = mjwarp.put_model(mjm)
    m.opt.ls_parallel = _LS_PARALLEL.value

    print(m.opt.solver)
    # m.opt.solver = mujoco.mjtSolver.mjSOL_CG
    # print(m.opt.solver)
    d = mjwarp.put_data(mjm, mjd)

    if _CLEAR_KERNEL_CACHE.value:
        wp.clear_kernel_cache()

    # wp.copy(d.ctrl, wp.array([mjd.ctrl.astype(np.float32)]))
    # wp.copy(d.act, wp.array([mjd.act.astype(np.float32)]))
    # wp.copy(d.xfrc_applied, wp.array([mjd.xfrc_applied.astype(np.float32)]))
    # wp.copy(d.qpos, wp.array([mjd.qpos.astype(np.float32)]))
    # wp.copy(d.qvel, wp.array([mjd.qvel.astype(np.float32)]))
    # Initialize input states
    # qpos_np = mjd.qpos.astype(np.float32)
    # qvel_np = mjd.qvel.astype(np.float32)
    # xpos_np = mjd.xpos.astype(np.float32)
    #
    # qpos = wp.array(qpos_np, dtype=wp.float32, requires_grad=True)
    # qvel = wp.array(qvel_np, dtype=wp.float32, requires_grad=True)
    # xpos = wp.array(xpos_np, dtype=wp.float32, requires_grad=True)
    #
    # target = wp.array([0.0, 0.0, 0.5], dtype=wp.float32, requires_grad=True)
    # # Copy into simulation data
    # wp.copy(d.qpos, qpos)
    # wp.copy(d.qvel, qvel)
    # wp.copy(d.xpos, xpos)
    # np_rand = np.random.rand(21).astype(np.float32)

    # wp_arr = wp.array(np_rand, requires_grad=True)


    # double warmup to work around issues with compilation during graph capture:
    print("before warmup")
    mjwarp.step(m, d)
    mjwarp.step(m, d)
    print("warmup done")
    tape = wp.Tape()


    print("is gpu", wp.get_device().is_cuda)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    with wp.ScopedCapture() as capture:
    # for _ in range(500):


        with tape: 
            mjwarp.step(m, d)
            wp.launch(
                    kernel=loss_kernel,
                    dim=1,
                    inputs=[d.xpos, loss]

            )
        tape.backward(loss)


    print("Doing backwards pass and calculating losses...")
    graph = capture.graph
    # with wp.Tape() as tape:
    for _ in range(500):
        with wp.ScopedTimer("step"):
            tape.zero()
            wp.capture_launch(graph)
        # wp.synchronize()


    # tape.backward(loss)
    # Print loss and gradient w.r.t. qpos
    print(f"Loss: {loss}")
    print("outputting gradients")
    print(tape.gradients)

    # Done
    print("Differentiation complete.")


def main():
    app.run(_main)


if __name__ == "__main__":
    main()

