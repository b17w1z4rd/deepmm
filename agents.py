"""
agents.py - market-making agents, including a neural policy trained by
reinforcement learning (policy gradient / REINFORCE) on the autodiff engine.

The environment (fills, price moves) is stochastic and NOT differentiable, so we
can't backprop through it as deep hedging does. Instead we use REINFORCE: roll
out episodes with a stochastic Gaussian policy, then push the policy toward
actions that led to above-average risk-adjusted reward, using the score-function
gradient  E[ (R - baseline) * grad log pi(a|s) ].  The autodiff engine supplies
grad log pi; the reward carries an inventory penalty so the maker learns to lean
against its position - which is exactly what Avellaneda-Stoikov does in closed
form.
"""
import sys
import numpy as np
from autodiff import Tensor
from market import PARAMS

sys.setrecursionlimit(1_000_000)


def softplus(x):
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))


class Adam:
    def __init__(self, params, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(t.data) for t in params]
        self.v = [np.zeros_like(t.data) for t in params]
        self.t = 0
    def zero_grad(self):
        for t in self.p:
            t.grad = np.zeros_like(t.data)
    def step(self):
        self.t += 1
        for i, t in enumerate(self.p):
            g = t.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            t.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


class NeuralMM:
    """Policy: state [inventory, time-remaining] -> mean raw quote offsets.
    Actual offsets are softplus(mean + noise), so they stay non-negative."""
    def __init__(self, hidden=32, q_scale=5.0, T=1.0, std=0.25, seed=0):
        rng = np.random.default_rng(seed)
        w = lambda a, b: Tensor(rng.standard_normal((a, b)) * np.sqrt(2.0 / a))
        self.W1, self.b1 = w(2, hidden), Tensor(np.zeros((1, hidden)))
        self.W2, self.b2 = w(hidden, hidden), Tensor(np.zeros((1, hidden)))
        self.W3, self.b3 = w(hidden, 2), Tensor(np.zeros((1, 2)))
        self.q_scale, self.T, self.std = q_scale, T, std

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def feat(self, q, tau):
        n = np.shape(q)[0] if np.ndim(q) else 1
        return np.stack([np.asarray(q) / self.q_scale,
                         np.full(n, tau) / self.T], axis=1)

    def mean_np(self, feat):
        z = np.tanh(feat @ self.W1.data + self.b1.data)
        z = np.tanh(z @ self.W2.data + self.b2.data)
        return z @ self.W3.data + self.b3.data

    def mean_tensor(self, feat):
        X = Tensor(feat)
        z = (X @ self.W1 + self.b1).tanh()
        z = (z @ self.W2 + self.b2).tanh()
        return z @ self.W3 + self.b3

    def quote_fn(self):                       # deterministic policy for evaluation
        def qf(S, q, tau):
            d = softplus(self.mean_np(self.feat(q, tau)))
            return d[:, 0], d[:, 1]
        return qf


def rollout(policy, M, p, eta, seed):
    """One batch of stochastic episodes. Records states/actions for the update."""
    rng = np.random.default_rng(seed)
    N, dt = p["N"], p["T"] / p["N"]
    A, kappa, sigma = p["A"], p["kappa"], p["sigma"]
    S = np.full(M, p["S0"]); q = np.zeros(M); cash = np.zeros(M); pen = np.zeros(M)
    feats = np.zeros((N, M, 2)); acts = np.zeros((N, M, 2))
    for t in range(N):
        tau = p["T"] - t * dt
        f = policy.feat(q, tau); feats[t] = f
        a = policy.mean_np(f) + policy.std * rng.standard_normal((M, 2)); acts[t] = a
        d = softplus(a); db, da = d[:, 0], d[:, 1]
        pb = 1 - np.exp(-A * np.exp(-kappa * db) * dt)
        pa = 1 - np.exp(-A * np.exp(-kappa * da) * dt)
        fb = (rng.random(M) < pb).astype(float)
        fa = (rng.random(M) < pa).astype(float)
        cash += fa * (S + da) - fb * (S - db); q += fb - fa
        pen += q * q * dt
        S = S + sigma * np.sqrt(dt) * rng.standard_normal(M)
    pnl = cash + q * S
    return pnl - eta * pen, pnl, feats, acts


def train(policy, p=PARAMS, eta=0.05, iters=150, M=512, lr=5e-3, log_every=25):
    opt = Adam(policy.params(), lr=lr)
    hist = []
    for it in range(1, iters + 1):
        R, pnl, feats, acts = rollout(policy, M, p, eta, seed=1000 + it)
        adv = (R - R.mean()) / (R.std() + 1e-8)          # normalized advantage
        logp = Tensor(np.zeros((M, 1)))
        for t in range(len(feats)):
            mu = policy.mean_tensor(feats[t])
            diff = (Tensor(acts[t]) - mu) * (1.0 / policy.std)
            logp = logp + (diff * diff).sum(axis=1, keepdims=True) * (-0.5)
        loss = -((Tensor(adv.reshape(M, 1))) * logp).mean()   # maximize E[adv*logpi]
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(R.mean()))
        if log_every and (it % log_every == 0 or it == 1):
            print(f"  iter {it:4d}   mean reward {R.mean():+.3f}   mean P&L {pnl.mean():+.3f}")
    return hist


if __name__ == "__main__":
    from market import simulate, fixed_quoter, as_quoter
    p = PARAMS
    avg = p["gamma"] * p["sigma"] ** 2 * (p["T"] / 2) + (2 / p["gamma"]) * np.log(1 + p["gamma"] / p["kappa"])

    pol = NeuralMM(seed=0)
    print("training neural market maker (REINFORCE) ...")
    train(pol, p, eta=1.0, iters=150, M=512, lr=5e-3)

    wf, qf = simulate(fixed_quoter(avg / 2), 40_000, p, seed=7)
    wa, qa = simulate(as_quoter(p), 40_000, p, seed=7)
    wn, qn = simulate(pol.quote_fn(), 40_000, p, seed=7)
    def row(n, w, q):
        print(f"{n:20s} {w.mean():+7.2f}   {w.std():6.2f}    {np.abs(q).std():5.2f}   "
              f"{np.quantile(-w,0.95):+7.2f}")
    print("\n                      mean     std    inv_std   CVaR95")
    row("fixed spread", wf, qf); row("Avellaneda-Stoikov", wa, qa); row("neural (RL)", wn, qn)
    assert np.abs(qn).std() < np.abs(qf).std(), "neural MM should control inventory"
    print("\nPASS: neural market maker learned inventory-aware quoting")
