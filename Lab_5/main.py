"""Лабораторна робота №5
Складова формула Сімпсона, методи Рунге-Ромберга, Ейткена, адаптивний алгоритм.
f(x) = 50 + 20 sin(pi x / 12) + 5 exp(-0.2 (x-12)^2), x in [0, 24]
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a, b = 0.0, 24.0


def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


# 2. Точне значення інтегралу (аналітично):
# int 50 dx = 50(b-a); int 20 sin(pi x/12) dx = 0 за повний період;
# int 5 exp(-0.2 (x-12)^2) dx = 5 * sqrt(pi/0.2) * erf(12*sqrt(0.2))
I0 = 50 * (b - a) + 5 * math.sqrt(math.pi / 0.2) * math.erf(12 * math.sqrt(0.2))


# 3. Складова формула Сімпсона (N - парне)
def simpson(N, func=f, a=a, b=b):
    if N % 2:
        N += 1
    h = (b - a) / N
    x = a + h * np.arange(N + 1)
    y = func(x)
    return h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])


if __name__ == "__main__":
    print(f"I0 = {I0:.12f}")

    # 4. Залежність похибки від N
    Ns = np.arange(10, 1001, 2)
    errs = np.array([abs(simpson(N) - I0) for N in Ns])
    eps = 1e-12
    idx = np.where(errs <= eps)[0]
    Nopt = int(Ns[idx[0]]) if len(idx) else int(Ns[np.argmin(errs)])
    epsopt = abs(simpson(Nopt) - I0)
    print(f"Nopt = {Nopt}, epsopt = {epsopt:.3e}")
    for N in (10, 20, 40, 80, 160, 320, 640, 1000):
        print(f"N={N:5d}  I={simpson(N):.12f}  err={abs(simpson(N)-I0):.3e}")

    # 5. N0 ~ Nopt/10, кратне 8
    N0 = max(8, int(round(Nopt / 10 / 8)) * 8)
    eps0 = abs(simpson(N0) - I0)
    print(f"N0 = {N0}, I(N0) = {simpson(N0):.12f}, eps0 = {eps0:.3e}")

    # 6. Рунге-Ромберг (p = 4, q = 2 -> 2^4 - 1 = 15)
    I1, I2, I4 = simpson(N0), simpson(N0 // 2), simpson(N0 // 4)
    IR = I1 + (I1 - I2) / 15
    epsR = abs(IR - I0)
    print(f"I(N0)={I1:.12f} I(N0/2)={I2:.12f} I(N0/4)={I4:.12f}")
    print(f"IR = {IR:.12f}, epsR = {epsR:.3e}")

    # 7. Ейткен
    IE = (I2 ** 2 - I1 * I4) / (2 * I2 - (I1 + I4))
    p = math.log(abs((I4 - I2) / (I2 - I1))) / math.log(2)
    epsE = abs(IE - I0)
    print(f"IE = {IE:.12f}, p = {p:.4f}, epsE = {epsE:.3e}")

    # 9. Адаптивний алгоритм
    calls = 0

    def fc(x):
        global calls
        calls += 1
        return float(f(x))

    def adaptive(fa, fm, fb, a_, b_, delta, depth=0):
        """Повертає (інтеграл, кількість відрізків). Умова (3): |I1 - I2| <= delta."""
        h = b_ - a_
        m = (a_ + b_) / 2
        I_1 = h / 6 * (fa + 4 * fm + fb)
        l, r = (a_ + m) / 2, (m + b_) / 2
        fl, fr = fc(l), fc(r)
        I_2 = h / 12 * (fa + 4 * fl + fm) + h / 12 * (fm + 4 * fr + fb)
        if abs(I_1 - I_2) <= max(delta, 1e-15 * abs(I_2)) or depth >= 25:
            return I_2, 1
        # delta залежить від кроку: на половинному відрізку - delta/2
        L, nl = adaptive(fa, fl, fm, a_, m, delta / 2, depth + 1)
        R, nr = adaptive(fm, fr, fb, m, b_, delta / 2, depth + 1)
        return L + R, nl + nr

    def run_adaptive(delta):
        global calls
        calls = 0
        fa, fb, fm = fc(a), fc(b), fc((a + b) / 2)
        val, n = adaptive(fa, fm, fb, a, b, delta)
        return val, abs(val - I0), calls, n

    deltas = [10.0 ** (-k) for k in range(1, 13)]
    print("delta, I, eps, calls, intervals")
    ad = []
    for d in deltas:
        v, e, c, n = run_adaptive(d)
        ad.append((d, v, e, c, n))
        print(f"{d:.0e}  {v:.12f}  {e:.3e}  {c}  {n}")

    # Для порівняння: скільки викликів f у звичайній Сімпсона з тією ж точністю
    # --- Графіки ---
    xs = np.linspace(a, b, 1000)
    plt.figure(figsize=(6.5, 4))
    plt.plot(xs, f(xs), label=r"$f(x)=50+20\sin(\pi x/12)+5e^{-0.2(x-12)^2}$")
    plt.xlabel("Час, x (год)"); plt.ylabel("Навантаження, f(x)")
    plt.grid(alpha=.4); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig("fig1_function.png", dpi=150); plt.close()

    plt.figure(figsize=(6.5, 4.3))
    plt.loglog(Ns, np.maximum(errs, 1e-17), lw=1)
    plt.axhline(eps, color="r", ls="--", label=f"ε = 1e-12")
    plt.axvline(Nopt, color="g", ls=":", label=f"N_opt = {Nopt}")
    plt.xlabel("N"); plt.ylabel("ε(N) = |I(N) − I₀|")
    plt.grid(True, which="both", alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig("fig2_error_vs_N.png", dpi=150); plt.close()

    plt.figure(figsize=(6.5, 4.3))
    names = ["Сімпсон\n(N0)", "Рунге–\nРомберг", "Ейткен"]
    vals = [eps0, epsR, epsE]
    plt.bar(names, vals, color=["#1F4E79", "#2E75B6", "#9DC3E6"])
    plt.yscale("log"); plt.ylabel("Абсолютна похибка")
    for i, v in enumerate(vals):
        plt.text(i, v * 1.4, f"{v:.1e}", ha="center", fontsize=9)
    plt.ylim(min(vals) / 10, max(vals) * 10); plt.grid(axis="y", alpha=.3)
    plt.savefig("fig3_comparison.png", dpi=150); plt.close()

    # Похибки методів залежно від N0
    N_list = np.arange(8, 321, 8)
    e_s, e_r, e_e = [], [], []
    for n in N_list:
        s1, s2, s4 = simpson(n), simpson(n // 2), simpson(n // 4)
        e_s.append(abs(s1 - I0))
        e_r.append(abs(s1 + (s1 - s2) / 15 - I0))
        den = 2 * s2 - (s1 + s4)
        e_e.append(abs((s2 ** 2 - s1 * s4) / den - I0) if den != 0 else np.nan)
    plt.figure(figsize=(6.5, 4.3))
    plt.loglog(N_list, np.maximum(e_s, 1e-17), label="Сімпсон")
    plt.loglog(N_list, np.maximum(e_r, 1e-17), label="Рунге–Ромберг")
    plt.loglog(N_list, np.maximum(e_e, 1e-17), label="Ейткен")
    plt.xlabel("N"); plt.ylabel("Абсолютна похибка")
    plt.grid(True, which="both", alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig("fig4_methods_vs_N.png", dpi=150); plt.close()

    dd = np.array([r[0] for r in ad]); ee = np.array([r[2] for r in ad]); cc = [r[3] for r in ad]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].loglog(dd, np.maximum(ee, 1e-17), "o-"); ax[0].set_xlabel("δ"); ax[0].set_ylabel("ε")
    ax[0].grid(True, which="both", alpha=.3); ax[0].invert_xaxis()
    ax[1].semilogx(dd, cc, "o-"); ax[1].set_xlabel("δ"); ax[1].set_ylabel("Кількість обчислень f")
    ax[1].grid(True, which="both", alpha=.3); ax[1].invert_xaxis()
    plt.tight_layout(); plt.savefig("fig5_adaptive.png", dpi=150); plt.close()

    ext = []
    print("N0, eps0, epsR, epsE, p")
    for n in (8, 16, 24, 32, 48, 64, 80, 160):
        s1, s2, s4 = simpson(n), simpson(n // 2), simpson(n // 4)
        r = s1 + (s1 - s2) / 15
        den = 2 * s2 - (s1 + s4)
        e = (s2 ** 2 - s1 * s4) / den
        pp = math.log(abs((s4 - s2) / (s2 - s1))) / math.log(2) if s2 != s1 else float('nan')
        ext.append((n, abs(s1 - I0), abs(r - I0), abs(e - I0), pp))
        print(n, *[f"{v:.3e}" for v in ext[-1][1:]])
    import json
    json.dump(dict(I0=I0, Nopt=Nopt, epsopt=epsopt, N0=N0, I1=I1, I2=I2, I4=I4, eps0=eps0,
                   IR=IR, epsR=epsR, IE=IE, p=p, epsE=epsE, ad=ad, ext=ext,
                   table=[(N, simpson(N), abs(simpson(N) - I0)) for N in (10, 20, 40, 80, 160, 320, 640, 1000)]),
              open("results.json", "w"))