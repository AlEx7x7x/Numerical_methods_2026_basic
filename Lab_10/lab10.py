# Лабораторна робота №10. Розв'язання задачі Коші: Адамс (прогноз-корекція) та Рунге-Кутта 4
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- ВАРІАНТ (змінити під свій) ----------
f = lambda x, y: x + y
exact = lambda x: 2*np.exp(x) - x - 1
a, b, y0 = 0.0, 1.0, 1.0
# -------------------------------------------------

# ===== Ч.2. Рунге-Кутта 4-го порядку =====
def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h/2*k1)
    k3 = f(x + h/2, y + h/2*k2)
    k4 = f(x + h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)

def rk4(h, a=a, b=b, y0=y0):
    n = int(round((b - a)/h))
    x = a + h*np.arange(n + 1)
    y = np.empty(n + 1); y[0] = y0
    for i in range(n):
        y[i+1] = rk4_step(x[i], y[i], h)
    return x, y

def rk4_error_exact(h):               # п.7: похибка за точним розв'язком
    x, y = rk4(h)
    return x, np.abs(y - exact(x))

def rk4_error_runge(h):               # п.8: оцінка за Рунге, R = |y_h/2 - y_h| / (2^4 - 1)
    x, y = rk4(h)
    x2, y2 = rk4(h/2)
    return x, np.abs(y2[::2] - y)/15

def rk4_auto(eps, h0=0.2):            # п.9: автоматичний вибір кроку
    x, y, h = a, y0, h0
    X, Y, H = [x], [y], []
    while x < b - 1e-12:
        h = min(h, b - x)
        while True:
            y1 = rk4_step(x, y, h)
            ym = rk4_step(x + h/2, rk4_step(x, y, h/2), h/2)
            R = abs(ym - y1)/15
            if R > eps: h /= 2; continue
            break
        x += h; y = ym
        X.append(x); Y.append(y); H.append(h)
        if R < eps/2**5: h *= 2       # eps/2^(s+1)
    return np.array(X), np.array(Y), np.array(H)

# ===== Ч.1. Адамс 2-го порядку: прогноз + модифікація + корекція =====
def adams2(h, a=a, b=b, y0=y0, tol=1e-12, itmax=2):
    n = int(round((b - a)/h))
    x = a + h*np.arange(n + 1)
    y = np.empty(n + 1); y[0] = y0
    y[1] = rk4_step(x[0], y[0], h)     # стартове значення методом РК4
    err = np.zeros(n + 1)              # оцінка похибки (y^c - y^p)/6
    dprev = 0.0                        # y_n^c - y_n^p з попереднього кроку
    for i in range(1, n):
        fn, fn1 = f(x[i], y[i]), f(x[i-1], y[i-1])
        yp = y[i] + h/2*(3*fn - fn1)                    # прогноз
        ym = yp + 5/6*dprev                             # модифікація
        yc = ym
        for _ in range(itmax):                          # корекція (ітерації)
            ynew = y[i] + h/2*(f(x[i+1], yc) + fn)
            if abs(ynew - yc) < tol: yc = ynew; break
            yc = ynew
        dprev = yc - yp
        y[i+1] = yc - dprev/6                           # кінцеве значення
        err[i+1] = abs(dprev)/6                         # оцінка локальної похибки
    return x, y, err

def adams_auto(eps, h0=0.2):          # п.5: автоматичний вибір кроку для Адамса
    x, y, h = a, y0, h0
    X, Y, H = [x], [y], []
    yprev = rk4_step(x, y, -h)        # y(x-h) методом РК4 (крок назад)
    dprev = 0.0
    while x < b - 1e-12:
        h = min(h, b - x)
        yprev = rk4_step(x, y, -h)    # перезапуск з поточним кроком
        fn, fn1 = f(x, y), f(x - h, yprev)
        yp = y + h/2*(3*fn - fn1)
        yc = yp + 5/6*dprev
        for _ in range(2): yc = y + h/2*(f(x + h, yc) + fn)
        R = abs(yc - yp)/6
        if R > eps: h /= 2; dprev = 0.0; continue
        dprev = yc - yp
        x += h; y = yc - dprev/6
        X.append(x); Y.append(y); H.append(h)
        if R < eps/2**3: h *= 2; dprev = 0.0
    return np.array(X), np.array(Y), np.array(H)

if __name__ == "__main__":
    # --- Ч.1 ---
    h = 0.1
    x, y, est = adams2(h)
    e_ex = np.abs(y - exact(x))
    plt.figure(); plt.semilogy(x[1:], e_ex[1:], "o-", label="за точним розв'язком")
    plt.semilogy(x[2:], est[2:], "s--", label="оцінка (y^к−y^п)/6")
    plt.xlabel("x"); plt.ylabel("похибка"); plt.title(f"Адамс 2-го порядку, h={h}")
    plt.legend(); plt.grid(True, which="both"); plt.savefig("adams_error.png", dpi=120)
    print("Адамс, h=0.1: max|y-y_точне| =", e_ex.max())
    for eps in (1e-3, 1e-4, 1e-5):
        X, Y, H = adams_auto(eps)
        print(f"Адамс авто eps={eps:g}: вузлів={len(X)}, h_min={H.min():.4g}, h_max={H.max():.4g}, "
              f"max похибка={np.abs(Y-exact(X)).max():.2e}")
    X, Y, H = adams_auto(1e-4)
    plt.figure(); plt.step(X[:-1], H, where="post"); plt.xlabel("x"); plt.ylabel("h")
    plt.title("Адамс: автоматичний крок, eps=1e-4"); plt.grid(True); plt.savefig("adams_step.png", dpi=120)

    # --- Ч.2 ---
    plt.figure()
    print("\nРК4: залежність похибки від кроку")
    prev = None
    for h in (0.2, 0.1, 0.05, 0.025):
        x, e = rk4_error_exact(h); _, r = rk4_error_runge(h)
        ratio = "" if prev is None else f"  (відношення {prev/e.max():.1f} ~ 16)"
        print(f"h={h:<6} max|err|={e.max():.3e}  max Runge={r.max():.3e}{ratio}")
        plt.semilogy(x[1:], e[1:], "o-", label=f"точна, h={h}")
        plt.semilogy(x[1:], r[1:], "x--", label=f"Рунге, h={h}")
        prev = e.max()
    plt.xlabel("x"); plt.ylabel("похибка"); plt.legend(fontsize=7); plt.grid(True, which="both")
    plt.title("Рунге-Кутта 4: локальна похибка"); plt.savefig("rk4_error.png", dpi=120)
    for eps in (1e-4, 1e-6, 1e-8, 1e-10):
        X, Y, H = rk4_auto(eps)
        print(f"РК4 авто eps={eps:g}: вузлів={len(X)}, h_min={H.min():.4g}, "
              f"max похибка={np.abs(Y-exact(X)).max():.2e}")
    X, Y, H = rk4_auto(1e-8)
    plt.figure(); plt.step(X[:-1], H, where="post"); plt.xlabel("x"); plt.ylabel("h")
    plt.title("РК4: автоматичний крок, eps=1e-8"); plt.grid(True); plt.savefig("rk4_step.png", dpi=120)
    # потрібний крок для заданої точності
    for eps in (1e-6, 1e-8):
        h = 0.2
        while rk4_error_runge(h)[1].max() > eps: h /= 2
        print(f"необхідний крок РК4 для eps={eps:g}: h={h:.4g}")
