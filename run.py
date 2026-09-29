"""
run.py - train the market makers and render the figure set.

Trains once, caches to figures/cache.npz, then draws seven dark-themed,
colorblind-safe figures. Re-run with --retrain to recompute.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from market import PARAMS, simulate, fixed_quoter, as_quoter, as_quotes
from agents import NeuralMM, train, softplus
from lob import OrderBook

# --- validated palette (dark surface) -------------------------------------- #
SURFACE, PAGE = "#1a1a19", "#0d0d0d"
INK, INK2, MUTED = "#ffffff", "#c3c2b7", "#898781"
GRID, AXIS = "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"

FIGDIR, CACHE = "figures", "figures/cache.npz"
P = PARAMS
ETAS = [0.25, 0.5, 1.0, 2.0, 4.0]
HEAD_ETA = 0.5


def set_style():
    plt.rcParams.update({
        "figure.facecolor": PAGE, "savefig.facecolor": PAGE,
        "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "font.family": "sans-serif", "font.size": 11,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})


def compute_all():
    os.makedirs(FIGDIR, exist_ok=True)
    avg = P["gamma"]*P["sigma"]**2*(P["T"]/2) + (2/P["gamma"])*np.log(1+P["gamma"]/P["kappa"])

    # frontier: train a neural MM at each inventory-penalty level
    fr_mean, fr_inv = [], []
    head_pol, head_hist = None, None
    for eta in ETAS:
        print(f"training neural MM (eta={eta}) ...")
        pol = NeuralMM(seed=0)
        hist = train(pol, P, eta=eta, iters=200, M=512, lr=5e-3, log_every=0)
        w, q = simulate(pol.quote_fn(), 40_000, P, seed=7)
        fr_mean.append(w.mean()); fr_inv.append(np.abs(q).std())
        if abs(eta - HEAD_ETA) < 1e-9:
            head_pol, head_hist = pol, hist

    # evaluate the three strategies (headline neural = HEAD_ETA)
    wf, qf = simulate(fixed_quoter(avg/2), 40_000, P, seed=7)
    wa, qa = simulate(as_quoter(P), 40_000, P, seed=7)
    wn, qn = simulate(head_pol.quote_fn(), 40_000, P, seed=7)

    # inventory sample paths (record a small batch)
    _, tf = simulate(fixed_quoter(avg/2), 400, P, seed=3, record=True)
    _, ta = simulate(as_quoter(P), 400, P, seed=3, record=True)
    _, tn = simulate(head_pol.quote_fn(), 400, P, seed=3, record=True)

    # one neural episode's quotes
    _, t1 = simulate(head_pol.quote_fn(), 1, P, seed=11, record=True)

    # learned quoting skew vs inventory (at tau = T/2), neural vs AS
    qgrid = np.linspace(-6, 6, 121)
    tau = P["T"]/2
    feat = head_pol.feat(qgrid, tau)
    d = softplus(head_pol.mean_np(feat))
    skew_neural = d[:, 0] - d[:, 1]                     # bid_off - ask_off
    db_as, da_as = as_quotes(np.full_like(qgrid, P["S0"]), qgrid, tau,
                             P["gamma"], P["sigma"], P["kappa"])
    skew_as = db_as - da_as

    # order-book depth snapshot (build a synthetic book around mid=100)
    ob = OrderBook()
    rng = np.random.default_rng(0)
    for lvl in range(1, 9):
        ob.add_limit("buy", 100 - lvl, int(20 + 30*rng.random() + 8*lvl))
        ob.add_limit("sell", 100 + lvl, int(20 + 30*rng.random() + 8*lvl))
    bid_px = np.array([100-l for l in range(1, 9)])
    ask_px = np.array([100+l for l in range(1, 9)])
    bid_dep = np.array([ob.depth("buy", p) for p in bid_px])
    ask_dep = np.array([ob.depth("sell", p) for p in ask_px])

    res = dict(wf=wf, wa=wa, wn=wn, qf=qf, qa=qa, qn=qn,
               fr_mean=np.array(fr_mean), fr_inv=np.array(fr_inv),
               etas=np.array(ETAS),
               as_pt=np.array([wa.mean(), np.abs(qa).std()]),
               fx_pt=np.array([wf.mean(), np.abs(qf).std()]),
               inv_f=tf["q"][:40], inv_a=ta["q"][:40], inv_n=tn["q"][:40],
               ep_S=t1["S"][0], ep_db=t1["db"][0], ep_da=t1["da"][0], ep_q=t1["q"][0],
               qgrid=qgrid, skew_neural=skew_neural, skew_as=skew_as,
               hist=np.array(head_hist),
               bid_px=bid_px, ask_px=ask_px, bid_dep=bid_dep, ask_dep=ask_dep)
    np.savez(CACHE, **res)
    return res


# ---- figures -------------------------------------------------------------- #
def fig_orderbook(R):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(R["bid_px"], R["bid_dep"], color=BLUE, width=0.8, label="bids")
    ax.bar(R["ask_px"], R["ask_dep"], color=ORANGE, width=0.8, label="asks")
    ax.axvline(100, color=MUTED, ls="--", lw=1.2)
    ax.text(100, ax.get_ylim()[1]*0.95, " mid", color=MUTED, va="top")
    ax.set_title("Limit order book depth  ·  price-time-priority engine")
    ax.set_xlabel("price"); ax.set_ylabel("resting size")
    ax.legend(frameon=False, labelcolor=INK2); ax.grid(axis="x", alpha=0)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/1_orderbook.png"); plt.close(fig)


def fig_training(R):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(R["hist"], color=BLUE, lw=2)
    ax.set_title("REINFORCE training  ·  mean reward per episode")
    ax.set_xlabel("iteration"); ax.set_ylabel("risk-adjusted reward")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/2_training.png"); plt.close(fig)


def fig_inventory(R):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    xs = np.linspace(0, P["T"], R["inv_f"].shape[1])
    for ax, inv, c, name in [(axes[0], R["inv_f"], MUTED, "fixed spread"),
                             (axes[1], R["inv_a"], ORANGE, "Avellaneda-Stoikov"),
                             (axes[2], R["inv_n"], BLUE, "neural (RL)")]:
        for i in range(inv.shape[0]):
            ax.plot(xs, inv[i], color=c, lw=0.6, alpha=0.4)
        ax.axhline(0, color=INK2, lw=1, ls="--")
        ax.set_title(name); ax.set_xlabel("time")
        ax.text(0.03, 0.96, f"|q| std = {np.abs(inv[:, -1]).std():.2f}",
                transform=ax.transAxes, va="top", color=INK2, fontsize=10)
    axes[0].set_ylabel("inventory  q")
    fig.suptitle("Inventory paths  ·  fixed spread drifts, AS and the RL agent lean back to zero",
                 y=1.02, color=INK, fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/3_inventory.png", bbox_inches="tight"); plt.close(fig)


def fig_pnl(R):
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.linspace(0, 100, 400)
    for w, c, name in [(R["wf"], MUTED, "fixed spread"), (R["wa"], ORANGE, "Avellaneda-Stoikov"),
                       (R["wn"], BLUE, "neural (RL)")]:
        k = gaussian_kde(np.clip(w, 0, 100)); y = k(xs)
        ax.fill_between(xs, y, color=c, alpha=0.15); ax.plot(xs, y, color=c, lw=2, label=name)
    ax.set_title("Terminal P&L distribution")
    ax.set_xlabel("terminal P&L per session"); ax.set_ylabel("density"); ax.set_yticks([])
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/4_pnl_distribution.png"); plt.close(fig)


def fig_frontier(R):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    # each blue dot is an independently trained RL agent at a different risk aversion
    ax.scatter(R["fr_inv"], R["fr_mean"], color=BLUE, s=90, zorder=5,
               label="neural agents (independently trained)")
    ax.scatter([R["as_pt"][1]], [R["as_pt"][0]], color=ORANGE, s=150, zorder=6,
               marker="*", label="Avellaneda-Stoikov (analytic)")
    ax.scatter([R["fx_pt"][1]], [R["fx_pt"][0]], color=MUTED, s=110, zorder=5,
               label="fixed spread")
    ax.axvspan(0, R["as_pt"][1], color=BLUE, alpha=0.05)
    ax.set_title("Risk–return  ·  model-free RL vs the analytic optimum")
    ax.set_xlabel("inventory risk   std(|q|)   (lower = safer)")
    ax.set_ylabel("mean P&L per session")
    ax.text(0.02, 0.03, "shaded: lower inventory risk than AS", transform=ax.transAxes,
            color=MUTED, fontsize=9)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper right")
    ax.set_xlim(0, 5.3)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/5_frontier.png"); plt.close(fig)


def fig_policy(R):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(R["qgrid"], R["skew_as"], color=ORANGE, lw=2.2, label="Avellaneda-Stoikov (analytic)")
    ax.plot(R["qgrid"], R["skew_neural"], color=BLUE, lw=2.2, label="neural (learned)")
    ax.axhline(0, color=MUTED, lw=1, ls="--"); ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_title("Learned quoting skew vs inventory  ·  both lean against the position")
    ax.set_xlabel("inventory  q"); ax.set_ylabel("quote skew  (bid offset − ask offset)")
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/6_policy.png"); plt.close(fig)


def fig_quotes(R):
    fig, ax = plt.subplots(figsize=(9.2, 5))
    xs = np.linspace(0, P["T"], len(R["ep_S"]))
    ax.plot(xs, R["ep_S"], color=INK2, lw=1.6, label="mid")
    xh = xs[:-1]
    ax.plot(xh, R["ep_S"][:-1] - R["ep_db"], color=BLUE, lw=1.2, alpha=0.9, label="neural bid")
    ax.plot(xh, R["ep_S"][:-1] + R["ep_da"], color=ORANGE, lw=1.2, alpha=0.9, label="neural ask")
    ax.fill_between(xh, R["ep_S"][:-1] - R["ep_db"], R["ep_S"][:-1] + R["ep_da"],
                    color=BLUE, alpha=0.08)
    ax.set_title("One session  ·  the neural maker's quotes tracking and skewing around mid")
    ax.set_xlabel("time"); ax.set_ylabel("price")
    ax.legend(frameon=False, labelcolor=INK2, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/7_quotes.png"); plt.close(fig)


def make_figures(R):
    set_style()
    fig_orderbook(R); fig_training(R); fig_inventory(R); fig_pnl(R)
    fig_frontier(R); fig_policy(R); fig_quotes(R)
    print(f"wrote figures to {FIGDIR}/")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    if "--retrain" in sys.argv or not os.path.exists(CACHE):
        R = compute_all()
    else:
        R = dict(np.load(CACHE)); print("loaded cache")
    make_figures(R)
    for n, w, q in [("fixed", R["wf"], R["qf"]), ("AS", R["wa"], R["qa"]), ("neural", R["wn"], R["qn"])]:
        print(f"{n:8s} mean {w.mean():+7.2f}  std {w.std():6.2f}  inv_std {np.abs(q).std():5.2f}")


if __name__ == "__main__":
    main()
