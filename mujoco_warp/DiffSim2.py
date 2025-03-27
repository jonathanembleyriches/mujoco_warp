import mujoco
import mujoco_warp as mjwarp
import warp as wp
import numpy as np

from absl import app, flags
from typing import Sequence

_MODEL_PATH = flags.DEFINE_string("mjcf", None, "Path to MJCF file", required=True)


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
        raise app.UsageError("Too many command-line arguments")

    # Load MuJoCo model
    mjm = mujoco.MjModel.from_xml_path(_MODEL_PATH.value)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    print("Putting model into Warp...")
    model = mjwarp.put_model(mjm)
    data = mjwarp.put_data(mjm, mjd)

    print("Running warmup...")
    mjwarp.step(model, data)
    mjwarp.step(model, data)

    print("Capturing graph...")
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

    with wp.ScopedCapture() as capture:
        with wp.Tape() as tape:
            mjwarp.step(model, data)
            wp.launch(loss_kernel, dim=1, inputs=[data.xpos, loss])
        tape.backward(loss)

    graph = capture.graph

    # Rerun the captured graph multiple times
    for i in range(100):
        loss.zero_()
        wp.capture_launch(graph)
        print(f"Step {i}, Loss: {loss.numpy()[0]}")

    print("Differentiation complete")

def main():
    app.run(_main)

if __name__ == "__main__":
    main()

