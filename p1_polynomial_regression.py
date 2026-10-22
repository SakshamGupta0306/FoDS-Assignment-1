import os
import numpy as np
import matplotlib.pyplot as plt
from common import (data_path, load_dataset, plot_true_sine,
                    generate_polynomial_features, write_report)

output_dir = "output part 1"
os.makedirs(output_dir, exist_ok=True)

SEED = 42          # seed for the per-epoch shuffling (stated in the report)
D = 9
BATCH_SIZE = 3


def loss(Q, a, t):
    """E(alpha) = 0.5 * ||Q alpha - t||^2 (unnormalised SSE)."""
    return 0.5 * np.sum((Q @ a - t) ** 2)


def batch_gd(Q, t, eta, epochs):
    a = np.zeros(Q.shape[1])
    losses = [loss(Q, a, t)]                      # epoch 0 = initial loss
    for _ in range(epochs):
        a = a - eta * (Q.T @ (Q @ a - t))         # full sum, not divided by N
        losses.append(loss(Q, a, t))
    return a, np.array(losses)


def sgd_and_minibatch(Q, t, eta, epochs, seed):
    """SGD and mini-batch GD (size 3, averaged gradient); same shuffle per epoch."""
    rng = np.random.default_rng(seed)
    N = Q.shape[0]
    a_sgd = np.zeros(Q.shape[1])
    a_mb = np.zeros(Q.shape[1])
    l_sgd = [loss(Q, a_sgd, t)]
    l_mb = [loss(Q, a_mb, t)]
    for _ in range(epochs):
        idx = rng.permutation(N)
        # SGD: N updates per epoch, single-point gradient contribution
        for i in idx:
            qi = Q[i:i + 1, :]
            a_sgd = a_sgd - eta * (qi.T @ (qi @ a_sgd - t[i:i + 1]))
        l_sgd.append(loss(Q, a_sgd, t))
        # Mini-batch: batches of 3,3,3,1 with AVERAGED gradient
        for s in range(0, N, BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            Qb, tb = Q[b, :], t[b]
            a_mb = a_mb - eta * (Qb.T @ (Qb @ a_mb - tb)) / len(b)
        l_mb.append(loss(Q, a_mb, t))
    return a_sgd, np.array(l_sgd), a_mb, np.array(l_mb)


def run_part1():
    x_train, y_train = load_dataset(data_path('poly_train.csv'))
    report = ["PART 1 REPORT", "=" * 60,
              f"Random seed used for shuffling: {SEED} (np.random.default_rng({SEED}))", ""]

    # ------------------------------------------------------------------
    # Deliverable 1: closed-form fits d = 1..9
    # ------------------------------------------------------------------
    x_plot = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Deliverable 1: Closed-Form Polynomial Fits (d=1 to 9)")
    report.append("Deliverable 1: closed-form (np.linalg.lstsq) fits")
    for d, ax in zip(range(1, 10), axes.ravel()):
        Phi = generate_polynomial_features(x_train, d)
        w, _, _, _ = np.linalg.lstsq(Phi, y_train, rcond=None)
        sse = 0.5 * np.sum((Phi @ w - y_train) ** 2)
        report.append(f"  d={d}: E={sse:.3e}, ||w||_2={np.linalg.norm(w):.3e}")
        ax.scatter(x_train, y_train, color='red', label='Train data')
        plot_true_sine(ax)
        ax.plot(x_plot, generate_polynomial_features(x_plot, d) @ w,
                color='blue', label=f'Fit d={d}')
        ax.set_title(f'Degree {d}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylim(-1.5, 1.5)
        ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deliverable_1_fits.png'), dpi=120)
    plt.close()
    report.append("")

    # ------------------------------------------------------------------
    # Setup for d = 9 (closed form + thin QR)
    # ------------------------------------------------------------------
    Phi = generate_polynomial_features(x_train, D)
    w_closed, _, _, _ = np.linalg.lstsq(Phi, y_train, rcond=None)
    Q, R = np.linalg.qr(Phi)                      # thin QR, Phi = Q R
    report.append(f"cond(Phi) for d=9: {np.linalg.cond(Phi):.3e}")
    report.append(f"||w_closed||_2 (d=9): {np.linalg.norm(w_closed):.6e}")
    report.append("")

    # ------------------------------------------------------------------
    # Deliverables 2 & 3: GD family, loss curves (eta = 0.1, 2000 epochs)
    # ------------------------------------------------------------------
    eta, epochs = 0.1, 2000
    a_batch, l_batch = batch_gd(Q, y_train, eta, epochs)
    a_sgd, l_sgd, a_mb, l_mb = sgd_and_minibatch(Q, y_train, eta, epochs, SEED)

    plt.figure(figsize=(8, 6))
    plt.plot(l_batch, label='Batch GD', linewidth=4, alpha=0.6)
    plt.plot(l_sgd, label='Stochastic GD', linestyle='--', linewidth=2)
    plt.plot(l_mb, label='Mini-batch GD (size 3)', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel(r'Objective $E(\alpha)$ (log scale)')
    plt.title(r'Deliverable 3: Loss Curves for GD Variants ($\eta=0.1$, d=9, QR basis)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(output_dir, 'deliverable_3_loss_curves.png'), dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Deliverable 4: learning-rate study (Batch GD, QR, d=9)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # Left: 100 epochs, the three required regimes (+ eta = 2 for reference)
    for eta_i, style in [(0.01, '-'), (0.67, '-'), (2.1, '-'), (2.0, ':')]:
        _, l = batch_gd(Q, y_train, eta_i, 100)
        axes[0].plot(l, linestyle=style, label=rf'$\eta={eta_i}$')
    axes[0].set_title('100 epochs: small / good / divergent (and $\\eta=2$)')
    # Right: longer range so that slow convergence at eta = 0.01 is visible
    for eta_i in [0.01, 0.1, 0.67, 2.0]:
        _, l = batch_gd(Q, y_train, eta_i, 1500)
        axes[1].plot(l, label=rf'$\eta={eta_i}$')
    axes[1].set_title('1500 epochs: slow vs. fast convergence')
    for ax in axes:
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'Objective $E(\alpha)$ (log scale)')
        ax.grid(True, which="both", ls="--")
        ax.legend()
    fig.suptitle('Deliverable 4: Batch GD Learning-Rate Study (d=9, QR basis)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deliverable_4_lr_study.png'), dpi=120)
    plt.close()

    report.append("Deliverable 4: learning-rate study (Batch GD, 100 epochs)")
    for eta_i, name in [(0.01, "too small"), (0.67, "well chosen")]:
        a_i, l = batch_gd(Q, y_train, eta_i, 100)
        report.append(f"  eta={eta_i} ({name}): E after 100 epochs = {l[-1]:.6e}")
    a_div, l_div = batch_gd(Q, y_train, 2.1, 100)
    w_div = np.linalg.solve(R, a_div)             # w = R^{-1} alpha
    report.append("  eta=2.1 (divergent): "
                  f"E after 100 epochs = {l_div[-1]:.6e}")
    report.append("  Weight vector w after 100 epochs (eta=2.1):")
    report.append("  " + np.array2string(w_div, precision=6, max_line_width=100))
    a_two, l_two = batch_gd(Q, y_train, 2.0, 101)
    report.append(
        f"  eta=2 exactly: E at epochs 98..101 = {l_two[98]:.6f}, {l_two[99]:.6f}, "
        f"{l_two[100]:.6f}, {l_two[101]:.6f}.")
    report.append(
        "  Because Q^T Q = I (largest Hessian eigenvalue 1), the error in each "
        "coordinate is multiplied by (1 - eta) every epoch. For eta=2 the factor "
        "is exactly -1: alpha flips between 0 and 2*alpha* and E stays constant "
        "(no convergence, no divergence). For eta>2, |1-eta|>1 so E grows "
        "geometrically; for eta=0.01 the factor 0.99 gives very slow decay.")
    report.append("")

    # ------------------------------------------------------------------
    # Deliverable 5: verification against closed form (same 2000 epochs)
    # ------------------------------------------------------------------
    norm_closed = np.linalg.norm(w_closed)
    report.append(f"Deliverable 5: verification (eta={eta}, {epochs} epochs, d=9)")
    report.append(f"  ||w_closed||_2 = {norm_closed:.6e}")
    for name, a in [('Batch GD', a_batch), ('Stochastic GD', a_sgd), ('Mini-batch GD', a_mb)]:
        w_gd = np.linalg.solve(R, a)
        abs_err = np.linalg.norm(w_gd - w_closed)
        report.append(f"  {name}: ||w_GD - w_closed||_2 = {abs_err:.6e}, "
                      f"relative error = {abs_err / norm_closed:.6e}")
    report.append(
        "  Note: the entries of w are very large for d=9 (||w_closed|| ~ 1e5), so an "
        "absolute error of ~1e-6 is tiny in relative terms. The three errors are "
        "almost identical because all variants have converged to machine precision; "
        "the remaining gap is the numerical floor from R^{-1} (cond(R) ~ 1e7) and "
        "lstsq, not a difference between the GD variants.")
    report.append("")

    # ------------------------------------------------------------------
    # Deliverable 6: discussion
    # ------------------------------------------------------------------
    report.append("Deliverable 6: Discussion")
    report.append(
        "In the loss-curve plot (eta=0.1), the SGD curve lies on top of Batch GD because "
        "Q is a square orthogonal matrix: its rows are orthonormal, so each single-row "
        "update changes only the component of alpha along that row and does not disturb "
        "the others. After one epoch every component has been multiplied by (1 - eta), "
        "exactly as in one Batch GD step, regardless of shuffle order. Mini-batch GD is "
        "slower because averaging divides the step by the batch size: the 9 samples in "
        "batches of 3 move with an effective step eta/3, so their components shrink by "
        "(1 - eta/3) per epoch. Only the singleton batch gets the full step, and the "
        "singleton is a random sample (1 in 10 per epoch), so the average log-rate is "
        "0.9*ln(1-eta/3) + 0.1*ln(1-eta) = -0.041 versus ln(0.9) = -0.105 for Batch GD "
        "(about 0.96 vs 0.90 per epoch). This ~2.6x slower rate matches the plot, where "
        "Mini-batch flattens at about epoch 850 versus about 330. SGD and Mini-batch "
        "share the same per-epoch shuffle.")

    write_report(os.path.join(output_dir, 'part1_report.txt'), report)


if __name__ == "__main__":
    run_part1()
    print(f"Part 1 complete. Check the '{output_dir}' directory for outputs.")