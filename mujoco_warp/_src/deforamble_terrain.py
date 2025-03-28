import warp as wp
import numpy as np
from warp.types import *
from warp.constants import _update_efc_row
@wp.struct
class TerrainParams:
    sigma_flat: float  # N/m^3
    sigma_cone: float  # N/m^3
    mu: float          # Friction coefficient
    k_h: float         # HSR stiffness
    b_h: float         # HSR damping
    beta_d: float      # Depth scaling
    rc: float          # Crater bottom radius
    gamma_c: float     # Crater angle (radians)


def randomize_terrain():
    # Randomization ranges from Table S3

    mu = np.random.uniform(0.3, 1.0)
    k_h = np.random.uniform(160.0, 1600.0)
    b_h = np.random.uniform(1.0, 6.0)

    sigma_flat = np.random.uniform(1.0e6, 3.0e6)
    sigma_cone = np.random.uniform(0.5e6, 1.5e6)

    # Soft perturbation (optional)
    sigma_flat += np.random.uniform(-1.0e6, 1.0e6)
    sigma_cone += np.random.uniform(-5.0e5, 5.0e5)

    # Threshold rejection
    if sigma_flat < 0.15e6:
        sigma_flat = 0.15e6
    if sigma_cone < 0.1e6:
        sigma_cone = 0.1e6

    # Fixed crater parameters
    rc = 0.05
    gamma_c = np.deg2rad(45.0)

    return TerrainParams(
        sigma_flat=float(sigma_flat),
        sigma_cone=float(sigma_cone),
        mu=float(mu),
        k_h=float(k_h),
        b_h=float(b_h),
        beta_d=0.5,  # as used in paper
        rc=float(rc),
        gamma_c=float(gamma_c),
    )


@wp.kernel
def deformable_contact_model(
    m: types.Model,
    d: types.Data,
    terrain_params: TerrainParams,
    penetration_buffer: float,
    p_max_array: wp.array(dtype=wp.vec3),
    z_max_array: wp.array(dtype=float),
):
    conid = wp.tid()

    if conid >= d.ncon[0]:
        return

    if d.contact.dim[conid] != 3:
        return

    # --- Penetration condition ---
    penetration = d.contact.dist[conid] - (d.contact.includemargin[conid] - penetration_buffer)
    if penetration >= 0.0:
        return

    efcid = wp.atomic_add(d.nefc, 0, 1)
    worldid = d.contact.worldid[conid]

    # --- Depth-dependent normal force ---
    area_sub = 0.01  # Example foot area
    area_cone = 0.02
    Fz = -(
        terrain_params.sigma_flat * area_sub +
        terrain_params.sigma_cone * area_cone
    )

    # --- Coulomb tangential friction (smoothed) ---
    mu = terrain_params.mu
    epsilon = 1e-3  # Smoothing parameter

    contact_frame = d.contact.frame[conid]
    foot_vel = wp.vec3(d.qvel[worldid, 0], d.qvel[worldid, 1], d.qvel[worldid, 2])

    tangent_dir = wp.vec3(contact_frame[1, 0], contact_frame[1, 1], contact_frame[1, 2])
    rel_vel = wp.dot(foot_vel, tangent_dir)

    mu_scaled = mu * wp.tanh(rel_vel / epsilon)
    tangential_force = mu_scaled * Fz * tangent_dir

    # --- HSR Force ---
    foot_pos = d.contact.pos[conid]
    p_max = p_max_array[conid]
    z_max = z_max_array[conid]

    d_HSR = wp.length(foot_pos - p_max) + (
        0.05 / wp.sin(terrain_params.gamma_c)
    ) - terrain_params.rc - (z_max - foot_pos[2] + 0.05) / wp.tan(terrain_params.gamma_c)

    planar_vel = wp.vec2(d.qvel[worldid, 0], d.qvel[worldid, 1])
    F_HSR = terrain_params.k_h * (d_HSR + terrain_params.beta_d * (z_max - foot_pos[2]))
    F_HSR += terrain_params.b_h * wp.length(planar_vel)

    # --- Apply Forces ---
    normal_jacobian = wp.vec3(0.0, 0.0, 1.0) * Fz
    hsr_force_vec = wp.vec3(F_HSR, 0.0, 0.0)  # Simplified in X direction

    d.efc.J[efcid, :] = normal_jacobian + tangential_force + hsr_force_vec

    # --- Update EFC row ---
    _update_efc_row(
        m, d, worldid, efcid, -penetration, -penetration, 1.0,
        d.contact.solref[conid], d.contact.solimp[conid],
        d.contact.includemargin[conid], True, 0.0
    )