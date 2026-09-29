"""
market.py - the market-making environment and the Avellaneda-Stoikov benchmark.

The mid-price is an arithmetic Brownian motion. A market maker posts a bid at
S - delta_b and an ask at S + delta_a; a quote gets hit by an incoming market
order with intensity lambda(delta) = A * exp(-kappa * delta) - tighter quotes
fill more often. Getting hit on the bid means we BUY (inventory up); on the ask
we SELL (inventory down). Inventory left on the book at the close is marked at
the final mid, so an unbalanced inventory is exposed to price risk.

Avellaneda-Stoikov (2008) solve for the risk-averse optimal quotes: they skew
around a *reservation price* that leans against inventory.
"""
import numpy as np

# canonical Avellaneda-Stoikov toy parameters
PARAMS = dict(S0=100.0, T=1.0, N=200, sigma=2.0, A=140.0, kappa=1.5, gamma=0.1)


def as_quotes(S, q, tau, gamma, sigma, kappa):
    """Avellaneda-Stoikov optimal (bid_offset, ask_offset) from the mid."""
    reservation = S - q * gamma * sigma ** 2 * tau
    spread = gamma * sigma ** 2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / kappa)
    db = S - (reservation - spread / 2)      # = spread/2 + q*gamma*sigma^2*tau
    da = (reservation + spread / 2) - S      # = spread/2 - q*gamma*sigma^2*tau
    return np.maximum(db, 0.0), np.maximum(da, 0.0)


def simulate(quote_fn, M, p=PARAMS, seed=0, record=False):
    """
    Vectorized market-making episodes.
      quote_fn(S, q, tau) -> (delta_b, delta_a), each shape (M,)
    Returns terminal wealth (M,), plus optional trajectories.
    """
    rng = np.random.default_rng(seed)
    N, dt = p["N"], p["T"] / p["N"]
    S = np.full(M, p["S0"]); q = np.zeros(M); cash = np.zeros(M)
    A, kappa, sigma = p["A"], p["kappa"], p["sigma"]

    if record:
        S_hist = np.zeros((M, N + 1)); q_hist = np.zeros((M, N + 1))
        db_hist = np.zeros((M, N)); da_hist = np.zeros((M, N))
        S_hist[:, 0] = S

    for t in range(N):
        tau = p["T"] - t * dt
        db, da = quote_fn(S, q, tau)
        db = np.maximum(db, 0.0); da = np.maximum(da, 0.0)
        pb = 1.0 - np.exp(-A * np.exp(-kappa * db) * dt)    # P(bid hit -> we buy)
        pa = 1.0 - np.exp(-A * np.exp(-kappa * da) * dt)    # P(ask hit -> we sell)
        fb = (rng.random(M) < pb).astype(float)
        fa = (rng.random(M) < pa).astype(float)
        cash += fa * (S + da) - fb * (S - db)
        q += fb - fa
        S = S + sigma * np.sqrt(dt) * rng.standard_normal(M)
        if record:
            S_hist[:, t + 1] = S; q_hist[:, t + 1] = q
            db_hist[:, t] = db; da_hist[:, t] = da

    wealth = cash + q * S
    if record:
        return wealth, dict(S=S_hist, q=q_hist, db=db_hist, da=da_hist, qT=q)
    return wealth, q


def fixed_quoter(half_spread):
    def qf(S, q, tau):
        n = np.shape(S)
        return np.full(n, half_spread), np.full(n, half_spread)
    return qf


def as_quoter(p=PARAMS):
    def qf(S, q, tau):
        return as_quotes(S, q, tau, p["gamma"], p["sigma"], p["kappa"])
    return qf


if __name__ == "__main__":
    p = PARAMS
    # Fixed-spread half-spread comparable to the AS average, for a fair contrast.
    avg_spread = p["gamma"] * p["sigma"] ** 2 * (p["T"] / 2) + \
                 (2 / p["gamma"]) * np.log(1 + p["gamma"] / p["kappa"])
    wf, qf = simulate(fixed_quoter(avg_spread / 2), 20_000, p, seed=1)
    wa, qa = simulate(as_quoter(p), 20_000, p, seed=1)

    print(f"                 mean P&L   std P&L   inventory std |q_T|")
    print(f"fixed spread     {wf.mean():+.3f}    {wf.std():6.3f}      {np.abs(qf).std():.2f}")
    print(f"Avellaneda-Stoikov {wa.mean():+.3f}    {wa.std():6.3f}      {np.abs(qa).std():.2f}")
    print(f"\nmean fills/episode ~ check both quote reasonably")
    assert np.abs(qa).std() < np.abs(qf).std(), "AS should hold less inventory risk"
    assert wa.std() < wf.std(), "AS should have tighter P&L"
    print("PASS: Avellaneda-Stoikov controls inventory risk vs a fixed spread")
