import torch
import numpy as np
import abc
from tqdm import trange

from src.utils import mask_adjs, mask_x, gen_noise
from src.sde.equations import VPSDE, subVPSDE, VESDE


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, adj, flags, t):
        dt = -1.0 / self.rsde.N
        noise = gen_noise(adj, flags)
        drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=True)
        adj_mean = adj + drift * dt
        adj = adj_mean + diffusion[:, None, None] * np.sqrt(-dt) * noise
        return adj, adj_mean


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, adj, flags, t):
        f, G = self.rsde.discretize(x, adj, flags, t, is_adj=True)
        noise = gen_noise(adj, flags)
        adj_mean = adj - f
        adj = adj_mean + G[:, None, None] * noise
        return adj, adj_mean


class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
        pass

    def update_fn(self, x, adj, flags, t):
        return adj, adj


class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__(sde, score_fn, snr, scale_eps, n_steps)

    def update_fn(self, x, adj, flags, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, adj, flags, t)
            noise = gen_noise(adj, flags)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            adj_mean = adj + step_size[:, None, None] * grad
            adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
        return adj, adj_mean


def get_score_fn(sde, model, train=True, continuous=True):

    if not train:
        model.eval()
    model_fn = model

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):

        def score_fn(x, adj, flags, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous:
                score = model_fn(x, adj, flags)
                std = sde.marginal_prob(torch.zeros_like(adj), t)[1]
            else:
                raise NotImplementedError(f"Discrete not supported")
            score = -score / std[:, None, None]
            return score

    elif isinstance(sde, VESDE):

        def score_fn(x, adj, flags, t):
            if continuous:
                score = model_fn(x, adj, flags)
            else:
                raise NotImplementedError(f"Discrete not supported")
            return score

    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

    return score_fn


def get_pc_solver(
    sde_adj,
    shape_x,
    shape_adj,
    predictor="Euler",
    corrector="None",
    snr=0.1,
    scale_eps=1.0,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
):

    def pc_solver(model_adj, init_flags):

        score_fn_adj = get_score_fn(
            sde_adj, model_adj, train=False, continuous=continuous
        )

        predictor_fn = (
            ReverseDiffusionPredictor
            if predictor == "Reverse"
            else EulerMaruyamaPredictor
        )
        corrector_fn = LangevinCorrector if corrector == "Langevin" else NoneCorrector

        predictor_obj_adj = predictor_fn(sde_adj, score_fn_adj, probability_flow)
        corrector_obj_adj = corrector_fn(sde_adj, score_fn_adj, snr, scale_eps, n_steps)

        with torch.no_grad():
            x = torch.ones(
                shape_x, device=device
            )  # TODO: Add logic here to replicate dataset
            adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
            flags = init_flags
            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)
            diff_steps = sde_adj.N
            timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

            # -------- Reverse diffusion process --------
            for i in trange(
                0, (diff_steps), desc="[Sampling]", position=1, leave=False
            ):
                t = timesteps[i]
                vec_t = torch.ones(shape_adj[0], device=t.device) * t
                adj, adj_mean = corrector_obj_adj.update_fn(x, adj, flags, vec_t)
                adj, adj_mean = predictor_obj_adj.update_fn(x, adj, flags, vec_t)
            print(" ")
            return (
                (adj_mean if denoise else adj),
                diff_steps * (n_steps + 1),
            )

    return pc_solver


def get_s4_solver(
    sde_adj,
    shape_x,
    shape_adj,
    predictor="None",
    corrector="None",
    snr=0.1,
    scale_eps=1.0,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
):

    def s4_solver(model_adj, init_flags):
        score_fn_adj = get_score_fn(
            sde_adj, model_adj, train=False, continuous=continuous
        )

        with torch.no_grad():
            x = torch.ones(
                shape_x, device=device
            )  # TODO: Add logic here to replicate dataset
            adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
            flags = init_flags
            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)
            diff_steps = sde_adj.N
            timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
            dt = -1.0 / diff_steps

            for i in trange(
                0, (diff_steps), desc="[Sampling]", position=1, leave=False
            ):
                t = timesteps[i]
                vec_t = torch.ones(shape_adj[0], device=t.device) * t
                vec_dt = torch.ones(shape_adj[0], device=t.device) * (dt / 2)

                score_adj = score_fn_adj(x, adj, flags, vec_t)
                Sdrift_adj = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

                # -------- Correction step --------
                timestep = (vec_t * (sde_adj.N - 1) / sde_adj.T).long()
                noise = gen_noise(adj, flags)
                grad_norm = torch.norm(
                    score_adj.reshape(score_adj.shape[0], -1), dim=-1
                ).mean()
                noise_norm = torch.norm(
                    noise.reshape(noise.shape[0], -1), dim=-1
                ).mean()
                if isinstance(sde_adj, VPSDE):
                    alpha = sde_adj.alphas.to(vec_t.device)[timestep]  # VP
                else:
                    alpha = torch.ones_like(vec_t)  # VE
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * score_adj
                adj = (
                    adj_mean
                    + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps
                )

                # -------- Prediction step --------
                adj_mean = adj
                mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt)
                adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

                adj = adj + Sdrift_adj * dt

                mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt)
                adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)
                adj_mean = mu_adj
            print(" ")
            return (adj_mean if denoise else adj), 0

    return s4_solver
