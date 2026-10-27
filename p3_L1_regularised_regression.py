import os
import numpy as np
import matplotlib.pyplot as plt
from common import data_path, load_dataset, plot_true_sine, calculate_rmse, write_report

output_dir = "output part 3"
os.makedirs(output_dir, exist_ok=True)

ETA = 0.05
ITERS = 20000
D = 9
E_STAR = 0.2625
W_STAR = np.array([0, -0.928, 0.045, 1.117, 0.032, -0.201, 0, -0.026, 0, 0])


def legendre_features(x, degree=D):
    """Shifted Legendre basis P_j(2x-1), j = 0..degree."""
    return np.polynomial.legendre.legvander(2 * x - 1, degree)


def lasso_obj(Phi, t, w, lam):
    r = Phi @ w - t
    return 0.5 * r @ r + lam * np.abs(w).sum()


def subgradient_descent(Phi, t, lam, eta=ETA, iters=ITERS):
    """Returns the FINAL iterate, E(w_k) for k=0..iters, and best-so-far E."""
    w = np.zeros(Phi.shape[1])
    E_hist = np.empty(iters + 1)
    for k in range(iters):
        r = Phi @ w - t
        E_hist[k] = 0.5 * r @ r + lam * np.abs(w).sum()
        w = w - eta * (Phi.T @ r + lam * np.sign(w))
    E_hist[iters] = lasso_obj(Phi, t, w, lam)
    best = np.minimum.accumulate(E_hist)           # used ONLY for the convergence plot
    return w, E_hist, best


def run_part3():
    x_train, y_train = load_dataset(data_path('poly_train.csv'))
    x_test, y_test = load_dataset(data_path('poly_test.csv'))
    Phi_train = legendre_features(x_train)
    Phi_test = legendre_features(x_test)
    lams = np.logspace(-4, 0, 21)
    report = ["PART 3 REPORT", "=" * 60, ""]

    # ------------------------------------------------------------------
    # Deliverable 1: convergence + verification at lambda = 0.1
    # ------------------------------------------------------------------
    lam = 0.1
    w, E_hist, best = subgradient_descent(Phi_train, y_train, lam)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, xscale, ttl in [(axes[0], 'linear', 'Linear iteration axis'),
                            (axes[1], 'log', 'Log iteration axis (shows the tail)')]:
        ax.plot(np.arange(len(E_hist)) + (1 if xscale == 'log' else 0), E_hist,
                color='orange', alpha=0.8, label='E(w)')
        ax.plot(np.arange(len(best)) + (1 if xscale == 'log' else 0), best,
                color='red', linestyle='--', linewidth=2, label='Best-so-far E')
        ax.axhline(E_STAR, color='k', linestyle=':', label=r'Reference $E^\star=0.2625$')
        ax.set_yscale('log')
        ax.set_xscale(xscale)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective E (log scale)')
        ax.set_title(ttl)
        ax.grid(True, which='both', ls='--')
        ax.legend()
    fig.suptitle(r'Deliverable 1: Lasso Subgradient Descent at $\lambda=0.1$')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deliverable_1_convergence.png'), dpi=120)
    plt.close()

    thr = 2 * ETA * lam
    zero = np.abs(w) < thr
    r = y_train - Phi_train @ w
    corr = Phi_train.T @ r
    nz_cond = np.max(np.abs(corr[~zero] - lam * np.sign(w[~zero]))) if (~zero).any() else 0.0
    z_cond = np.max(np.abs(corr[zero]) / lam) if zero.any() else 0.0
    E_final = lasso_obj(Phi_train, y_train, w, lam)
    report += [
        "Deliverable 1: lambda = 0.1 (eta=0.05, 20000 iterations, w0=0)",
        f"  Final-iterate E            = {E_final:.6f}",
        f"  Best-so-far E              = {best[-1]:.6f}",
        f"  Reference E*               = {E_STAR}",
        f"  E_final - E*               = {E_final - E_STAR:.3e}",
        f"  max |Phi_j^T r - lam*sign(w_j)| over non-~0 coeffs = {nz_cond:.6f}",
        f"  max |Phi_j^T r| / lam over ~0 coeffs (should be <= 1) = {z_cond:.6f}",
        f"  ||w - w*||_2               = {np.linalg.norm(w - W_STAR):.4f}",
        f"  (~0 rule |w_j| < 2*eta*lam = {thr:.3f}; coefficients chatter with "
        "amplitude ~ eta*lam, so agreement with w* is only to ~1e-2.)",
        ""]

    # ------------------------------------------------------------------
    # Sweep over lambda (final iterates only)
    # ------------------------------------------------------------------
    w_all, tr, te, zc = [], [], [], []
    for lam_i in lams:
        w_i, _, _ = subgradient_descent(Phi_train, y_train, lam_i)
        w_all.append(w_i)
        tr.append(calculate_rmse(y_train, Phi_train @ w_i))
        te.append(calculate_rmse(y_test, Phi_test @ w_i))
        zc.append(int(np.sum(np.abs(w_i) < 2 * ETA * lam_i)))
    w_all, tr, te, zc = map(np.array, (w_all, tr, te, zc))

    # Deliverable 2: RMSE sweep
    plt.figure(figsize=(10, 6))
    plt.plot(lams, tr, label='Train RMSE', marker='o')
    plt.plot(lams, te, label='Test RMSE', marker='s')
    plt.axhline(0.1, color='k', linestyle='--', label=r'Noise level $\sqrt{0.01}=0.1$')
    plt.xscale('log')
    plt.xlabel(r'$\lambda$ (log scale)')
    plt.ylabel('RMSE')
    plt.title(r'Deliverable 2: Lasso Train/Test RMSE vs. $\lambda$')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.savefig(os.path.join(output_dir, 'deliverable_2_rmse_sweep.png'), dpi=120)
    plt.close()

    # Deliverable 3: coefficient path + count of ~0 coefficients
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2.2, 1]})
    for j in range(D + 1):
        axes[0].plot(lams, w_all[:, j], label=f'$w_{j}$')
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$\lambda$ (log scale)')
    axes[0].set_ylabel('Coefficient value')
    axes[0].set_title('Lasso coefficient path (Legendre basis)')
    axes[0].grid(True, which='both', ls='--', alpha=0.5)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].step(lams, zc, where='mid', marker='o')
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r'$\lambda$ (log scale)')
    axes[1].set_ylabel(r'# coefficients with $|w_j|<2\eta\lambda$')
    axes[1].set_title(r'Number of $\approx 0$ coefficients')
    axes[1].grid(True, which='both', ls='--', alpha=0.5)
    fig.suptitle('Deliverable 3: Lasso Coefficient Path')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deliverable_3_coeff_path.png'), dpi=120)
    plt.close()

    report.append("Deliverable 3: number of ~0 coefficients (|w_j| < 2*eta*lambda) per lambda")
    report.append("  lambda      #~0   train RMSE  test RMSE")
    for lam_i, c, a, b_ in zip(lams, zc, tr, te):
        report.append(f"  {lam_i:.4e}  {c:3d}   {a:.6f}   {b_:.6f}")
    report.append("")

    # ------------------------------------------------------------------
    # Deliverable 4: lambda*
    # ------------------------------------------------------------------
    i_l = int(np.argmin(te))
    lam_l, w_l = lams[i_l], w_all[i_l]
    zero_idx = np.where(np.abs(w_l) < 2 * ETA * lam_l)[0]
    report += [
        "Deliverable 4: Lasso lambda*",
        f"  lambda* = {lam_l:.4f}, test RMSE = {te[i_l]:.6f}",
        f"  # ~0 coefficients = {len(zero_idx)}, indices = {zero_idx.tolist()}",
        ""]

    # ------------------------------------------------------------------
    # Deliverable 5: Ridge in the same Legendre basis, same grid
    # ------------------------------------------------------------------
    G = Phi_train.T @ Phi_train
    b = Phi_train.T @ y_train
    w_r_all, te_r = [], []
    for lam_i in lams:
        w_r = np.linalg.solve(G + lam_i * np.eye(D + 1), b)
        w_r_all.append(w_r)
        te_r.append(calculate_rmse(y_test, Phi_test @ w_r))
    te_r = np.array(te_r)
    i_r = int(np.argmin(te_r))
    lam_r, w_r = lams[i_r], w_r_all[i_r]
    edge = " (WARNING: at grid edge)" if i_r in (0, len(lams) - 1) else ""

    n_small_l = int(np.sum(np.abs(w_l) < 0.01))
    n_small_r = int(np.sum(np.abs(w_r) < 0.01))
    report += [
        "Deliverable 5: Lasso vs Ridge (each at its own best lambda by test RMSE)",
        "  Method | lambda* | Test RMSE | ||w||_2 | #(|w_j|<0.01)",
        "  -------|---------|-----------|---------|--------------",
        f"  Lasso  | {lam_l:.4f}  | {te[i_l]:.4f}    | {np.linalg.norm(w_l):.4f}  | {n_small_l}",
        f"  Ridge  | {lam_r:.4f}  | {te_r[i_r]:.4f}    | {np.linalg.norm(w_r):.4f}  | {n_small_r}{edge}",
        ""]

    # ------------------------------------------------------------------
    # Deliverable 6: fitted curves
    # ------------------------------------------------------------------
    x_plot = np.linspace(0, 1, 200)
    Phi_plot = legendre_features(x_plot)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='red', label='Train data')
    plot_true_sine(plt.gca())
    plt.plot(x_plot, Phi_plot @ w_l, color='blue', label=rf'Lasso ($\lambda^*={lam_l:.4f}$)')
    plt.plot(x_plot, Phi_plot @ w_r, color='green', linestyle='-.',
             label=rf'Ridge ($\lambda^*={lam_r:.4f}$)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-1.5, 1.5)
    plt.title(r"Deliverable 6: Lasso vs. Ridge Fit at Each Method's Best $\lambda$")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'deliverable_6_fitted_curves.png'), dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Deliverable 7: discussion (numbers filled in from the results)
    # ------------------------------------------------------------------
    report += [
        "Deliverable 7: Discussion",
        f"In the coefficient path, the L1 penalty drives coefficients to ~0 one by one "
        f"as lambda grows ({zc[0]} are ~0 at lambda={lams[0]:.0e}, {len(zero_idx)} at "
        f"lambda*={lam_l:.2f}, {zc[-1]} at lambda={lams[-1]:.0f}). The |w| penalty has a "
        "kink at 0, so its pull on w_j keeps magnitude lambda however small w_j gets, and "
        "w_j can sit at 0 whenever |Phi_j^T r| <= lambda (the L1 ball has corners on the "
        "axes); with a constant step the subgradient method only keeps it ~0, within "
        "about eta*lambda. The L2 pull, lambda*w_j, vanishes as w_j -> 0, so ridge only "
        "shrinks. "
        f"At their best lambda, Lasso has {n_small_l} coefficients with |w_j|<0.01 versus "
        f"{n_small_r} for Ridge, at similar ||w||_2 ({np.linalg.norm(w_l):.2f} vs "
        f"{np.linalg.norm(w_r):.2f}) and lower test RMSE ({te[i_l]:.4f} vs {te_r[i_r]:.4f}). "
        f"Past lambda*, train RMSE rises from {tr[i_l]:.3f} to {tr[-1]:.3f} and test RMSE "
        f"from {te[i_l]:.3f} to {te[-1]:.3f}; the two converge because the model is "
        "underfitting both sets."]

    write_report(os.path.join(output_dir, 'part3_report.txt'), report)


if __name__ == "__main__":
    run_part3()
    print(f"Part 3 complete. Check the '{output_dir}' directory for outputs.")