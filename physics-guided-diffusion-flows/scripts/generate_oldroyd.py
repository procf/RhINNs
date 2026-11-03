"""
Physics-guided diffusion sampling for the Oldroyd-B (viscoelastic) flow past a triangle.

Adapted from scripts/generate_ns_bounded.py of DiffusionPDE (Huang et al., 2024).
Changes: triangular obstacle detected from the reference final velocity; four-channel
state (u0, v0, uT, vT) normalized to [-1, 1]; continuity residual written on the
normalized, anisotropic grid (coefficient K); command-line overrides.

Run from the repository root:
    python -m scripts.generate_oldroyd --config configs/oldroyd-forward.yaml
"""

import tqdm
import pickle
import numpy as np
import torch
import dnnlib
import torch.nn.functional as F
from torch_utils import distributed as dist
import scipy.io
import os

# ----------------------------------------------------------------------------
# Triangle obstacle masks
# ----------------------------------------------------------------------------
def detect_triangle_mask(data_np, eps=1e-10, device=torch.device('cuda')):
    """Auto-detect the triangle obstacle from a raw (un-normalized) GT sample.

    The Oldroyd-B Triangle dataset has no-slip walls on the triangle, so pixels
    inside / on its boundary are exactly zero for both final-velocity channels.
    Verified across all 31 test files: same 42 pixels (y in [58,69], x in [32,37]).
    """
    uT = data_np[..., 2]
    vT = data_np[..., 3]
    mask = ((np.abs(uT) < eps) & (np.abs(vT) < eps)).astype(np.float32)
    return torch.tensor(mask, device=device)


def random_index_and_triangle(triangle_mask, k, grid_size, seed=0,
                              device=torch.device('cuda')):
    """k fraction of random pixels, plus the full triangle obstacle."""
    rng = np.random.default_rng(seed)
    rand_mask = torch.tensor(
        (rng.random((grid_size, grid_size)) < k).astype(np.float32),
        device=device,
    )
    mask = torch.maximum(rand_mask, triangle_mask.to(device))
    return mask, mask.sum().item()


def triangle_index(triangle_mask, device=torch.device('cuda')):
    """Just the triangle obstacle (no random observation points)."""
    mask = triangle_mask.to(device).to(torch.float32)
    return mask, mask.sum().item()


# ----------------------------------------------------------------------------
def continuity_coefficient(config):
    """Weight K of the dv/dy term in the continuity residual on the normalized grid.

    The 128x128 grid samples the physical domain [x_min, x_max] x [y_min, y_max]
    (columns along x, rows along y), and u, v are normalized to [-1, 1] with
    different ranges. Substituting both mappings into du/dx + dv/dy = 0 and
    dividing by the u-term scale gives d(u~)/dj + K d(v~)/di = 0, with
    K = (v-range / u-range) * (dx / dy).
    """
    u_range = config['generate']['u_max'] - config['generate']['u_min']
    v_range = config['generate']['v_max'] - config['generate']['v_min']
    x_len   = config['data']['x_max'] - config['data']['x_min']
    y_len   = config['data']['y_max'] - config['data']['y_min']
    # Same number of grid points in x and y, so dx / dy = x_len / y_len.
    return (v_range / u_range) * (x_len / y_len)


def get_ns_bounded_loss(a, u, a_GT, u_GT, a_mask, u_mask, k_cont=1.0, device=None):
    if device is None:
        device = a.device

    a    = a.to(torch.float64)
    u    = u.to(torch.float64)
    a_GT = a_GT.to(torch.float64)
    u_GT = u_GT.to(torch.float64)

    B, C, H, W = a.shape

    deriv_x = torch.tensor([[-1, 0, 1]], dtype=torch.float64, device=device).view(1, 1, 1, 3) / 2.0
    deriv_y = torch.tensor([[-1], [0], [1]], dtype=torch.float64, device=device).view(1, 1, 3, 1) / 2.0

    # Incompressibility on final velocity:  d(uT)/dx + d(vT)/dy = 0
    u_x = u[:, 0:1, :, :]
    u_y = u[:, 1:2, :, :]

    du_dx = F.conv2d(u_x, deriv_x, padding=(0, 1))
    dv_dy = F.conv2d(u_y, deriv_y, padding=(1, 0))

    # Continuity on the normalized square grid (see continuity_coefficient).
    pde_loss = du_dx + k_cont * dv_dy

    pde_loss[:, :, 0, :]  = 0.0
    pde_loss[:, :, -1, :] = 0.0
    pde_loss[:, :, :, 0]  = 0.0
    pde_loss[:, :, :, -1] = 0.0

    L_pde = torch.mean(pde_loss**2)

    if a_GT.dim() == 3:
        a_GT = a_GT.unsqueeze(0)
    if u_GT.dim() == 3:
        u_GT = u_GT.unsqueeze(0)

    if a_GT.shape[0] == 1 and B > 1:
        a_GT = a_GT.expand(B, -1, -1, -1)
    if u_GT.shape[0] == 1 and B > 1:
        u_GT = u_GT.expand(B, -1, -1, -1)

    if a_mask.dim() == 2:
        a_mask = a_mask.unsqueeze(0).unsqueeze(0)
    elif a_mask.dim() == 3:
        a_mask = a_mask.unsqueeze(1)

    if u_mask.dim() == 2:
        u_mask = u_mask.unsqueeze(0).unsqueeze(0)
    elif u_mask.dim() == 3:
        u_mask = u_mask.unsqueeze(1)

    a_mask = a_mask.to(torch.float64).to(device)
    u_mask = u_mask.to(torch.float64).to(device)

    if a_mask.shape[0] == 1 and B > 1:
        a_mask = a_mask.expand(B, -1, -1, -1)
    if u_mask.shape[0] == 1 and B > 1:
        u_mask = u_mask.expand(B, -1, -1, -1)

    err_a_sq        = (a - a_GT)**2
    err_a_sq_masked = err_a_sq * a_mask
    num_obs_a       = a_mask.sum() * C
    L_obs_a         = err_a_sq_masked.sum() / (num_obs_a + 1e-8)

    err_u_sq        = (u - u_GT)**2
    err_u_sq_masked = err_u_sq * u_mask
    num_obs_u       = u_mask.sum() * C
    L_obs_u         = err_u_sq_masked.sum() / (num_obs_u + 1e-8)

    return pde_loss.squeeze(1), L_pde, L_obs_a, L_obs_u

# ----------------------------------------------------------------------------
def normalize_uv_4ch(x_phys, u_min, u_max, v_min, v_max):
    """Normalize physical values to [-1, 1]."""
    x_norm = x_phys.clone()
    x_norm[:, 0, :, :] = 2.0 * (x_phys[:, 0, :, :] - u_min) / (u_max - u_min) - 1.0  # u0
    x_norm[:, 1, :, :] = 2.0 * (x_phys[:, 1, :, :] - v_min) / (v_max - v_min) - 1.0  # v0
    x_norm[:, 2, :, :] = 2.0 * (x_phys[:, 2, :, :] - u_min) / (u_max - u_min) - 1.0  # uT
    x_norm[:, 3, :, :] = 2.0 * (x_phys[:, 3, :, :] - v_min) / (v_max - v_min) - 1.0  # vT
    return x_norm

def unnormalize_uv_4ch(x4, u_min, u_max, v_min, v_max):
    """Unnormalize from [-1, 1] to physical values."""
    x_phys = x4.clone()
    x_phys[:, 0, :, :] = u_min + (x4[:, 0, :, :] + 1.0) * 0.5 * (u_max - u_min)  # u0
    x_phys[:, 1, :, :] = v_min + (x4[:, 1, :, :] + 1.0) * 0.5 * (v_max - v_min)  # v0
    x_phys[:, 2, :, :] = u_min + (x4[:, 2, :, :] + 1.0) * 0.5 * (u_max - u_min)  # uT
    x_phys[:, 3, :, :] = v_min + (x4[:, 3, :, :] + 1.0) * 0.5 * (v_max - v_min)  # vT
    return x_phys

# ----------------------------------------------------------------------------
def generate_ns_bounded(config):
    """Generate samples for Oldroyd-B flow over a triangle."""
    datapath   = config['data']['datapath']
    device     = config['generate']['device']
    base_name  = os.path.basename(datapath)
    sample_tag = os.path.splitext(base_name)[0]

    u_min = config['generate']['u_min']
    u_max = config['generate']['u_max']
    v_min = config['generate']['v_min']
    v_max = config['generate']['v_max']

    # Load test data — raw physical values (unnormalized test set, see triangle_data.ipynb)
    data_np = np.load(datapath)
    assert data_np.ndim == 3 and data_np.shape[2] == 4, \
        f"Expected (H,W,4), got {data_np.shape}"
    H, W, C = data_np.shape

    data_torch = torch.tensor(data_np, dtype=torch.float64, device=device)
    data_chw   = data_torch.permute(2, 0, 1)            # [4, H, W] physical

    a_GT = data_chw[0:2, :, :]    # (u0, v0) physical
    u_GT = data_chw[2:4, :, :]    # (uT, vT) physical

    # Normalize GT to [-1,1] for use in loss against x_cur (also in [-1,1])
    data_4ch_phys = data_chw.unsqueeze(0)
    data_4ch_norm = normalize_uv_4ch(data_4ch_phys, u_min, u_max, v_min, v_max)
    a_GT_norm = data_4ch_norm[0, 0:2, :, :]
    u_GT_norm = data_4ch_norm[0, 2:4, :, :]

    # Triangle obstacle — auto-detected from GT (pixels where uT, vT == 0).
    # All 31 test files share the same 42 pixels (y in [58,69], x in [32,37]).
    triangle_mask = detect_triangle_mask(data_np, device=device)
    print(f"Triangle obstacle pixels: {int(triangle_mask.sum().item())}")

    batch_size = config['generate']['batch_size']
    seed       = config['generate']['seed']
    torch.manual_seed(seed)

    zeta_obs_a_cfg = config['generate']['zeta_obs_a']
    za_str  = str(zeta_obs_a_cfg).replace('.', 'p')
    run_tag = f"{sample_tag}_seed{seed}_za{za_str}{config['generate'].get('tag', '')}"

    k_cont = continuity_coefficient(config)
    print(f"Continuity coefficient K = {k_cont:.4f}")

    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)

    print(f'Generating {batch_size} samples...')
    latents = torch.randn(
        [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
        device=device
    )

    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[
            torch.randint(net.label_dim, size=[batch_size], device=device)
        ]

    sigma_min = max(config['generate']['sigma_min'], net.sigma_min)
    sigma_max = min(config['generate']['sigma_max'], net.sigma_max)
    num_steps    = config['test']['iterations']
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    rho          = config['generate']['rho']

    sigma_t_steps = (sigma_max ** (1.0 / rho)
                     + step_indices / (num_steps - 1)
                     * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps),
                               torch.zeros_like(sigma_t_steps[:1])])

    x_next = latents.to(torch.float64) * sigma_t_steps[0]

    problem_type = config['test']['problem_type']
    if problem_type == 'both':
        known_index_a, _ = random_index_and_triangle(triangle_mask, 0.01, H, seed=1, device=device)
        known_index_u, _ = random_index_and_triangle(triangle_mask, 0.01, H, seed=0, device=device)
    elif problem_type == 'forward':
        # Given full initial condition + triangle BC on final state.
        known_index_a, _ = random_index_and_triangle(triangle_mask, 1.0, H, seed=1, device=device)
        known_index_u, _ = triangle_index(triangle_mask, device=device)
    elif problem_type == 'inverse':
        known_index_a, _ = triangle_index(triangle_mask, device=device)
        known_index_u, _ = random_index_and_triangle(triangle_mask, 0.01, H, seed=0, device=device)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    loss_L_obs_a, loss_L_obs_u, loss_L_pde = [], [], []
    grad_norm_obs_a, grad_norm_obs_u, grad_norm_pde = [], [], []

    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(
        list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))),
        unit='step'
    ):
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)

        x_N    = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur  = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

        if i < num_steps - 1:
            x_N    = net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
            d_prime = (x_next - x_N) / sigma_t_next
            x_next  = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

        # Losses in normalized space (same space as x_cur / x_N)
        a_N_norm = x_N[:, 0:2, :, :]    # [B,2,H,W] (u0, v0) normalized
        u_N_norm = x_N[:, 2:4, :, :]    # [B,2,H,W] (uT, vT) normalized

        pde_residual, L_pde, L_obs_a, L_obs_u = get_ns_bounded_loss(
            a_N_norm, u_N_norm, a_GT_norm, u_GT_norm,
            known_index_a, known_index_u, k_cont=k_cont, device=device
        )

        grad_x_cur_obs_a = torch.autograd.grad(outputs=L_obs_a, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_obs_u = torch.autograd.grad(outputs=L_obs_u, inputs=x_cur, retain_graph=True)[0]
        grad_x_cur_pde   = torch.autograd.grad(outputs=L_pde,   inputs=x_cur)[0]

        loss_L_obs_a.append(L_obs_a.detach().cpu().item())
        loss_L_obs_u.append(L_obs_u.detach().cpu().item())
        loss_L_pde.append(L_pde.detach().cpu().item())
        grad_norm_obs_a.append(grad_x_cur_obs_a.detach().cpu().norm().item())
        grad_norm_obs_u.append(grad_x_cur_obs_u.detach().cpu().norm().item())
        grad_norm_pde.append(grad_x_cur_pde.detach().cpu().norm().item())

        zeta_obs_a = config['generate']['zeta_obs_a']
        zeta_obs_u = config['generate']['zeta_obs_u']
        zeta_pde   = config['generate']['zeta_pde']

        if i <= 0.8 * num_steps:
            x_next = x_next - zeta_obs_a * grad_x_cur_obs_a - zeta_obs_u * grad_x_cur_obs_u
        else:
            x_next = x_next - 0.1 * (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u) - zeta_pde * grad_x_cur_pde

        if i % 100 == 0:
            diff_norm = ((sigma_t_next - sigma_t) * d_cur).norm().item()
            obs_step  = (zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u)
            pde_step  = zeta_pde * grad_x_cur_pde
            print(
                f"Step {i}",
                "||diff||", diff_norm,
                "||obs||",  obs_step.norm().item(),
                "||pde||",  pde_step.norm().item(),
            )

    outdir = config["generate"].get("outdir", ".")
    os.makedirs(outdir, exist_ok=True)
    log_name = os.path.join(outdir, f"ns_bounded_logs_tri_{run_tag}.npz")
    np.savez(
        log_name,
        L_obs_a=np.array(loss_L_obs_a),
        L_obs_u=np.array(loss_L_obs_u),
        L_pde=np.array(loss_L_pde),
        grad_norm_obs_a=np.array(grad_norm_obs_a),
        grad_norm_obs_u=np.array(grad_norm_obs_u),
        grad_norm_pde=np.array(grad_norm_pde),
        sigma_t_steps=sigma_t_steps.detach().cpu().numpy()
    )
    print(f"Saved denoising logs to {log_name}")

    # Final output: unnormalize x_next to physical values
    x_final = x_next
    x_phys  = unnormalize_uv_4ch(x_final, u_min, u_max, v_min, v_max)
    a_final = x_phys[:, 0:2, :, :]   # [B,2,H,W] physical (u0, v0)
    u_final = x_phys[:, 2:4, :, :]   # [B,2,H,W] physical (uT, vT)

    rel_err_a = torch.norm(a_final - a_GT.unsqueeze(0), p=2) / torch.norm(a_GT, p=2)
    rel_err_u = torch.norm(u_final - u_GT.unsqueeze(0), p=2) / torch.norm(u_GT, p=2)
    print(f'Relative error of a (IC):    {rel_err_a}')
    print(f'Relative error of u (final): {rel_err_u}')
    print("IC RMSE:", torch.sqrt(torch.mean((a_final - a_GT.unsqueeze(0))**2)).item())

    outdir = config["generate"].get("outdir", ".")
    os.makedirs(outdir, exist_ok=True)
    out_name = os.path.join(outdir, f"ns_bounded_uv_results_tri_{run_tag}.mat")
    scipy.io.savemat(out_name, {
        'a': a_final.detach().cpu().numpy(),
        'u': u_final.detach().cpu().numpy(),
    })
    print(f"Saved results to {out_name}")
    print("Final L_obs_a:", L_obs_a.item())
    print("Final L_obs_u:", L_obs_u.item())
    print("Final L_pde  :", L_pde.item())


# ----------------------------------------------------------------------------
def main():
    import argparse
    import yaml
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--config", required=True, help="Path to the YAML config.")
    p.add_argument("--datapath", default=None, help="Override data.datapath (one test sample, .npy).")
    p.add_argument("--pretrained", default=None, help="Override test.pre-trained (network pickle).")
    p.add_argument("--iterations", type=int, default=None, help="Override test.iterations (sampling steps).")
    p.add_argument("--batch-size", type=int, default=None, help="Override generate.batch_size.")
    p.add_argument("--zeta-obs-a", type=float, default=None, help="Override generate.zeta_obs_a.")
    p.add_argument("--seed", type=int, default=None, help="Override generate.seed.")
    p.add_argument("--tag", default=None, help="Override generate.tag (suffix of output file names).")
    p.add_argument("--outdir", default=None, help="Override generate.outdir (where outputs are written).")
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    overrides = {
        ("data", "datapath"): args.datapath,
        ("test", "pre-trained"): args.pretrained,
        ("test", "iterations"): args.iterations,
        ("generate", "batch_size"): args.batch_size,
        ("generate", "seed"): args.seed,
        ("generate", "tag"): args.tag,
        ("generate", "outdir"): args.outdir,
    }
    if args.zeta_obs_a is not None:
        # keep integer weights as ints so file names read za500, not za500p0
        za = args.zeta_obs_a
        overrides[("generate", "zeta_obs_a")] = int(za) if za.is_integer() else za
    for (section, key), value in overrides.items():
        if value is not None:
            cfg.setdefault(section, {})[key] = value

    generate_ns_bounded(cfg)
    print("Done.")


if __name__ == "__main__":
    main()
