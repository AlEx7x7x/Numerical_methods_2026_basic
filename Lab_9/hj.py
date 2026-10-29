# -*- coding: utf-8 -*-
"""Лабораторна робота №9. Метод Хука-Дживса багатовимірної оптимізації."""
import math

# ---------- Цільові функції для тестування ----------
def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

def wood(x):
    x1, x2, x3, x4 = x
    return (100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2 + 90 * (x4 - x3**2) ** 2 + (1 - x3) ** 2
            + 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2) + 19.8 * (x2 - 1) * (x4 - 1))

# ---------- Система нелінійних рівнянь (m = 2) ----------
def f1(x): return x[0] ** 2 + x[1] ** 2 - 4
def f2(x): return math.exp(x[0]) + x[1] - 1
def F(x):  # цільова функція: сума квадратів нев'язок
    return f1(x) ** 2 + f2(x) ** 2

# ---------- Метод Хука-Дживса ----------
def explore(fun, x, h, h_min, alpha, shrink):
    """Досліджуючий пошук з базисної точки x. Крок зменшується (shrink=True) або ні."""
    x = list(x); fx = fun(x); h = list(h)
    for i in range(len(x)):
        while True:
            xp = x[:]; xp[i] += h[i]; fp = fun(xp)
            if fp < fx: x, fx = xp, fp; break
            xm = x[:]; xm[i] -= h[i]; fm = fun(xm)
            if fm < fx: x, fx = xm, fm; break
            if not shrink: break
            h[i] /= alpha
            if h[i] < h_min: break
    return x, fx, h

def hooke_jeeves(fun, x0, h0=0.5, alpha=2.0, eps1=1e-8, eps2=1e-12, lam=1.0,
                 max_iter=100000, traj_file=None):
    """Повертає (x*, F(x*), число кроків, траєкторія)."""
    n = len(x0)
    base = list(x0); fb = fun(base)
    traj = [(base[:], fb)]
    for it in range(1, max_iter + 1):
        h = [h0] * n
        cur, fc, h = explore(fun, base, h, eps1, alpha, True)
        if cur == base:                       # зменшення кроку не дало руху -> мінімум
            break
        while True:                           # пошук по зразку
            if (math.sqrt(sum((a - b) ** 2 for a, b in zip(cur, base))) < eps1
                    and abs(fc - fb) < eps2):
                base, fb = cur, fc; traj.append((base[:], fb))
                return finish(base, fb, traj, traj_file)
            y = [c + lam * (c - b) for c, b in zip(cur, base)]
            z, fz, _ = explore(fun, y, [h0] * n, eps1, alpha, False)
            base, fb = cur, fc; traj.append((base[:], fb))
            if fz < fb:
                cur, fc = z, fz
            else:
                break                         # повертаємось до досліджуючого пошуку з base
    return finish(base, fb, traj, traj_file)

def finish(x, fx, traj, traj_file):
    if traj_file:
        with open(traj_file, "w") as fo:
            fo.write("k\t" + "\t".join(f"x{i+1}" for i in range(len(x))) + "\tF(x)\n")
            for k, (p, v) in enumerate(traj):
                fo.write(f"{k}\t" + "\t".join(f"{c:.10f}" for c in p) + f"\t{v:.6e}\n")
    return x, fx, len(traj) - 1, traj

if __name__ == "__main__":
    out = []
    P = lambda *s: (print(*s), out.append(" ".join(map(str, s))))
    P("=== Тест: функція Розенброка, x0=(-1.2; 1) ===")
    x, v, k, _ = hooke_jeeves(rosenbrock, [-1.2, 1.0], traj_file="traj_rosenbrock.txt")
    P(f"x* = ({x[0]:.8f}; {x[1]:.8f}), f = {v:.2e}, базисних точок: {k}")
    P("=== Тест: функція Вуда, x0=(-3; -1; -3; -1) ===")
    x, v, k, _ = hooke_jeeves(wood, [-3, -1, -3, -1], traj_file="traj_wood.txt")
    P("x* = (" + "; ".join(f"{c:.6f}" for c in x) + f"), f = {v:.2e}, базисних точок: {k}")
    P("=== Система: x1^2+x2^2=4, exp(x1)+x2=1 ===")
    for x0 in ([1.0, -1.5], [-1.5, 1.0]):
        x, v, k, _ = hooke_jeeves(F, x0, traj_file=f"traj_system_{'1' if x0[0]>0 else '2'}.txt")
        P(f"x0={x0}: x* = ({x[0]:.8f}; {x[1]:.8f}), F = {v:.2e}, f1 = {f1(x):.2e}, f2 = {f2(x):.2e}, базисних точок: {k}")
    open("results.txt", "w").write("\n".join(out))
