"""Classical fourth-order Runge–Kutta; no embedded error estimator."""

import jax.numpy as jnp


def rk4_step(rhs, x, y, h, args=(), *, return_stages=False):
    """Advance y'=rhs(x,y,args); usable inside jit, scan and while_loop.

    Optional stages include the final candidate, for domain checks by the caller.
    A domain-limited RHS must itself guard invalid intermediate states.
    """
    k1 = rhs(x, y, args)
    y2 = y + h * k1 / 2
    k2 = rhs(x + h / 2, y2, args)
    y3 = y + h * k2 / 2
    k3 = rhs(x + h / 2, y3, args)
    y4 = y + h * k3
    k4 = rhs(x + h, y4, args)
    candidate = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if return_stages:
        return candidate, jnp.stack((y, y2, y3, y4, candidate))
    return candidate
