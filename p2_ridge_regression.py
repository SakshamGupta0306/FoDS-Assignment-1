import os
import numpy as np
import matplotlib.pyplot as plt
from common import (data_path, load_dataset, plot_true_sine,
                    generate_polynomial_features, calculate_rmse, write_report)

output_dir = "output part 2"
os.makedirs(output_dir, exist_ok=True)


def ridge_objective(Phi, t, w, lam):
    return 0.5 * np.sum((Phi @ w - t) ** 2) + 0.5 * lam * np.sum(w ** 2)


def run_part2():
    x_train, y_train = load_dataset(data_path('poly_train.csv'))
    x_test, y_test = load_dataset(data_path('poly_test.csv'))

    d = 9
    Phi_train = generate_polynomial_features(x_train, d)
    Phi_test = generate_polynomial_features(x_test, d)
    lams = np.logspace(-10, 2, 49)
    I = np.eye(d + 1)
    G = Phi_train.T @ Phi_train
    b = Phi_train.T @ y_train
    report = ["PART 2 REPORT", "=" * 60, ""]

    # ------------------------------------------------------------------
    # Deliverable 1: closed form for every lambda (linear solve, no inverse)
    # ------------------------------------------------------------------
    w_all, train_rmse, test_rmse = [], [], []
    for lam in lams:
        w = np.linalg.solve(G + lam * I, b)
        w_all.append(w)
        train_rmse.append(calculate_rmse(y_train, Phi_train @ w))
        test_rmse.append(calculate_rmse(y_test, Phi_test @ w))
    w_all = np.array(w_all)
    train_rmse = np.array(train_rmse)
    test_rmse = np.array(test_rmse)

    # ------------------------------------------------------------------
    # Deliverable 2: GD agreement at lambda = 1e-2 (original basis, eta=0.04)
    # ------------------------------------------------------------------
    lam_gd, eta = 1e-2, 0.04
    A_gd = G + lam_gd * I
    w_ref = np.linalg.solve(A_gd, b)
    E_ref = ridge_objective(Phi_train, y_train, w_ref, lam_gd)
    norm_ref = np.linalg.norm(w_ref)

    w = np.zeros(d + 1)
    losses = [ridge_objective(Phi_train, y_train, w, lam_gd)]
    iters, rel_err, max_iter = 0, 1.0, 5_000_000
    while rel_err >= 1e-6 and iters < max_iter:
        grad = Phi_train.T @ (Phi_train @ w - y_train) + lam_gd * w
        w = w - eta * grad
        iters += 1
        losses.append(ridge_objective(Phi_train, y_train, w, lam_gd))
        rel_err = np.linalg.norm(w - w_ref) / norm_ref
    losses = np.array(losses)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(losses, color='purple', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel(r'Ridge objective $\tilde{E}(w)$')
    axes[0].set_title(r'$\tilde{E}(w)$ vs. iteration ($\lambda=10^{-2}$)')
    axes[0].grid(True, ls='--')
    gap = np.maximum(losses - E_ref, 1e-18)
    axes[1].plot(gap, color='darkgreen', linewidth=2)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel(r'$\tilde{E}(w_k)-\tilde{E}(w_{closed})$ (log scale)')
    axes[1].set_title('Gap to closed-form optimum')
    axes[1].grid(True, which='both', ls='--')
    fig.suptitle(r'Deliverable 2: GD Convergence at $\lambda = 10^{-2}$')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deliverable_2_gd_convergence.png'), dpi=120)
    plt.close()

    cond_plain = np.linalg.cond(G)
    cond_pen = np.linalg.cond(A_gd)
    report += [
        "Deliverable 2: GD agreement at lambda = 1e-2 (eta = 0.04, start w = 0)",
        f"  Iterations until ||w_GD - w_closed||/||w_closed|| < 1e-6: {iters}",
        f"  Final relative error: {rel_err:.3e}",
        f"  cond(Phi^T Phi)           = {cond_plain:.3e}",
        f"  cond(Phi^T Phi + lam*I)   = {cond_pen:.3e}",
        "  Conditioning: adding lambda*I lifts every eigenvalue by lambda, so the "
        "smallest eigenvalue is no longer ~0 and the condition number drops from "
        f"~{cond_plain:.1e} to ~{cond_pen:.1e}, which makes plain GD practical.",
        ""]

    # ------------------------------------------------------------------
    # Deliverable 3: RMSE sweep and lambda*
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(lams, train_rmse, label='Train RMSE', marker='o', markersize=4)
    plt.plot(lams, test_rmse, label='Test RMSE', marker='s', markersize=4)
    plt.axhline(0.1, color='black', linestyle='--', label=r'Noise level $\sqrt{0.01}=0.1$')
    plt.xscale('log')
    plt.xlabel(r'$\lambda$ (log scale)')
    plt.ylabel('RMSE')
    plt.title(r'Deliverable 3: Train/Test RMSE vs. $\lambda$ (ridge, d=9)')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.savefig(os.path.join(output_dir, 'deliverable_3_rmse_sweep.png'), dpi=120)
    plt.close()

    opt = int(np.argmin(test_rmse))
    lam_star, min_rmse = lams[opt], test_rmse[opt]
    mask = test_rmse <= 1.05 * min_rmse
    lo = hi = opt                                  # contiguous run around the minimum
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    while hi < len(lams) - 1 and mask[hi + 1]:
        hi += 1
    contiguous = (lams[lo], lams[hi])
    all_ok = lams[mask]
    is_contiguous = int(mask.sum()) == (hi - lo + 1)
    decades = np.log10(contiguous[1] / contiguous[0])

    report += [
        "Deliverable 3: lambda selection",
        f"  lambda* = {lam_star:.4e}",
        f"  Test RMSE at lambda* = {min_rmse:.6f} (train RMSE there = {train_rmse[opt]:.6f})",
        f"  1.05 x min test RMSE = {1.05 * min_rmse:.6f}",
        f"  Contiguous interval around lambda* with test RMSE <= 1.05 x min: "
        f"[{contiguous[0]:.4e}, {contiguous[1]:.4e}] ({decades:.1f} decades)",
        f"  Set of ALL grid lambdas satisfying the threshold is contiguous: {is_contiguous}",
    ]
    if not is_contiguous:
        report.append("  All grid lambdas satisfying it: " +
                      ", ".join(f"{v:.2e}" for v in all_ok))
    i_hi = int(np.argmin(np.abs(lams - 1e-2)))
    report += [
        f"  Test RMSE at lambda=1e-10: {test_rmse[0]:.4f}; at lambda=1e-2: "
        f"{test_rmse[i_hi]:.4f} (train {train_rmse[i_hi]:.4f})",
        "  Discussion: The minimum is flat rather than sharp, since test RMSE stays "
        f"within 5% of its minimum across about {decades:.1f} decades of lambda "
        f"([{contiguous[0]:.1e}, {contiguous[1]:.1e}]). Below the interval the fit "
        f"overfits (wiggly, large weights, test RMSE rising to {test_rmse[0]:.3f} "
        "while train RMSE is near 0), and above it the fit is over-smoothed so "
        "both train and test RMSE climb.",
        ""]

    # ------------------------------------------------------------------
    # Deliverable 4: fitted curves (overfit / lambda* / 1e-2)
    # ------------------------------------------------------------------
    x_plot = np.linspace(0, 1, 200)
    Phi_plot = generate_polynomial_features(x_plot, d)
    targets = [1e-10, lam_star, 1e-2]
    titles = [r"Overfit ($\lambda=10^{-10}$)",
              rf"Optimal ($\lambda^*={lam_star:.2e}$)",
              r"Underfit ($\lambda=10^{-2}$)"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(r"Deliverable 4: Effect of $\lambda$ on Model Fit")
    report.append("Deliverable 4: ||w||_2 of the three fits")
    for ax, t_lam, title in zip(axes, targets, titles):
        idx = int(np.argmin(np.abs(np.log10(lams) - np.log10(t_lam))))
        w = w_all[idx]
        report.append(f"  lambda = {lams[idx]:.3e}: ||w||_2 = {np.linalg.norm(w):.4f}")
        ax.scatter(x_train, y_train, color='red', label='Train data')
        plot_true_sine(ax)
        ax.plot(x_plot, Phi_plot @ w, color='blue', label='Ridge fit')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylim(-1.5, 1.5)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deliverable_4_fitted_curves.png'), dpi=120)
    plt.close()

    write_report(os.path.join(output_dir, 'part2_report.txt'), report)


if __name__ == "__main__":
    run_part2()
    print(f"Part 2 complete. Check the '{output_dir}' directory for outputs.")