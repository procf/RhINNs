"""Two-stage training (Section 2.2).

1. ``pretrain`` - Adam (or RMSprop) followed by L-BFGS on
   residual_weight*loss_R + data_weight*loss_D + l1_weight*||lambda||_1,
   with the library coefficients trained jointly with the network.
2. ``ado`` - alternating direction optimisation: STRidge updates lambda from
   the current network, then Adam refines the network with lambda frozen.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize
import tensorflow as tf

from .stridge import train_stridge


@dataclass
class PretrainConfig:
    rounds: int
    adam_steps: int
    residual_weight: float
    data_weight: float
    learning_rate: float
    optimizer: str = "adam"          # "adam" or "rmsprop"
    l1_weight: float = 1e-5


@dataclass
class ADOConfig:
    rounds: int
    adam_steps: int
    residual_weight: float
    data_weight: float
    stridge_d_tol: float = 1.0
    stridge: dict = field(default_factory=lambda: dict(
        lam=1e-5, maxit=10, str_iters=100, normalize=2, split=0.8))
    # Learning rate 5e-4, halved every 500 steps (staircase).
    learning_rate: float = 5e-4
    decay_steps: int = 500
    decay_rate: float = 0.5


def _gradient_fn(model, problem, variables, residual_weight, data_weight, l1_weight):
    @tf.function
    def fn():
        with tf.GradientTape() as tape:
            f, _, _ = problem.residual(model)
            loss_r = residual_weight * tf.reduce_mean(tf.square(f))
            loss_d = data_weight * problem.data_loss(model)
            loss_l1 = l1_weight * tf.norm(model.lambda1, ord=1)
            loss = loss_r + loss_d + loss_l1
        return loss, tape.gradient(loss, variables), (loss_d, loss_r, loss_l1)
    return fn


def _lbfgs(variables, gradient_fn, log_every=1000):
    """Minimise with scipy's L-BFGS-B over the flattened ``variables``."""
    shapes = [v.shape for v in variables]
    sizes = [int(np.prod(s)) for s in shapes]
    step = {"n": 0}

    def assign(x):
        for v, chunk, shape in zip(variables, np.split(x, np.cumsum(sizes)[:-1]), shapes):
            v.assign(tf.reshape(tf.cast(chunk, tf.float32), shape))

    def loss_and_grad(x):
        assign(x)
        loss, grads, parts = gradient_fn()
        step["n"] += 1
        if step["n"] % log_every == 0:
            _log(f"L-BFGS {step['n']}", loss, parts)
        flat = np.concatenate([g.numpy().ravel() for g in grads]).astype(np.float64)
        return loss.numpy().astype(np.float64), flat

    x0 = np.concatenate([v.numpy().ravel() for v in variables])
    scipy.optimize.minimize(loss_and_grad, x0, jac=True, method="L-BFGS-B")


def _log(prefix, loss, parts):
    loss_d, loss_r, loss_l1 = (float(p) for p in parts)
    print(f"{prefix}: loss {float(loss):.4e} | data {loss_d:.4e} | "
          f"residual {loss_r:.4e} | L1 {loss_l1:.4e}")


def pretrain(model, problem, cfg: PretrainConfig, log_every=1000):
    variables = model.trainable_variables
    grad_fn = _gradient_fn(model, problem, variables, cfg.residual_weight,
                           cfg.data_weight, cfg.l1_weight)
    opt_cls = {"adam": tf.keras.optimizers.legacy.Adam,
               "rmsprop": tf.keras.optimizers.legacy.RMSprop}[cfg.optimizer]
    optimizer = opt_cls(learning_rate=cfg.learning_rate)
    for r in range(cfg.rounds):
        for i in range(cfg.adam_steps):
            loss, grads, parts = grad_fn()
            optimizer.apply_gradients(zip(grads, variables))
            if i % log_every == 0:
                _log(f"pretrain round {r} step {i}", loss, parts)
        _lbfgs(variables, grad_fn)


def stridge_update(model, problem, cfg: ADOConfig):
    _, phi, deriv = problem.residual(model)
    lam = train_stridge(phi.numpy(), deriv.numpy(), d_tol=cfg.stridge_d_tol,
                        w_init=model.lambda1.numpy(), **cfg.stridge)
    model.lambda1.assign(lam.astype(np.float32))


def ado(model, problem, cfg: ADOConfig, log_every=100):
    variables = model.network_variables
    grad_fn = _gradient_fn(model, problem, variables, cfg.residual_weight,
                           cfg.data_weight, 0.0)
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        cfg.learning_rate, cfg.decay_steps, cfg.decay_rate, staircase=True)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=schedule)
    for r in range(cfg.rounds):
        stridge_update(model, problem, cfg)
        for i in range(cfg.adam_steps):
            loss, grads, parts = grad_fn()
            optimizer.apply_gradients(zip(grads, variables))
            if i % log_every == 0:
                _log(f"ADO round {r} step {i}", loss, parts)
    stridge_update(model, problem, cfg)
    return model.lambda1.numpy().ravel()
