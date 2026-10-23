# -*- coding: utf-8 -*-
"""Лабораторна робота №8. Чисельні методи розв'язування нелінійних рівнянь."""
import math, cmath

EPS = 1e-8

# ---------- Трансцендентна функція ----------
f   = lambda x: math.cos(x) - 0.1 * x
df  = lambda x: -math.sin(x) - 0.1
d2f = lambda x: -math.cos(x)

# ---------- 1. Табуляція ----------
def tabulate(a=0.0, b=10.0, h=0.1, fname="tab.txt"):
    n = int(round((b - a) / h))
    xs = [a + i * h for i in range(n + 1)]
    ys = [f(x) for x in xs]
    with open(fname, "w") as fo:
        fo.write("x\tf(x)\n")
        for x, y in zip(xs, ys):
            fo.write(f"{x:.4f}\t{y:.8f}\n")
    roots = []
    for i in range(n):
        if ys[i] * ys[i + 1] < 0:
            x0 = xs[i] - ys[i] * h / (ys[i + 1] - ys[i])
            roots.append((x0, "зростання" if ys[i + 1] > ys[i] else "спадання"))
    return roots

def done(xn, xp, eps):
    return abs(xn - xp) < eps and abs(f(xn)) < eps

# ---------- 2. Методи ----------
def simple_iter(x0, eps=EPS, N=10000):
    tau = -1.0 / df(x0)                      # метод релаксації: phi = x + tau*f(x)
    x = x0
    for k in range(1, N + 1):
        xn = x + tau * f(x)
        if done(xn, x, eps): return xn, k
        x = xn
    return x, N

def newton(x0, eps=EPS, N=1000):
    x = x0
    for k in range(1, N + 1):
        xn = x - f(x) / df(x)
        if done(xn, x, eps): return xn, k
        x = xn
    return x, N

def chebyshev(x0, eps=EPS, N=1000):
    x = x0
    for k in range(1, N + 1):
        fx, d1, d2 = f(x), df(x), d2f(x)
        xn = x - fx / d1 - d2 * fx**2 / (2 * d1**3)
        if done(xn, x, eps): return xn, k
        x = xn
    return x, N

def chord(x0, x1, eps=EPS, N=1000):
    xp, x = x0, x1
    for k in range(1, N + 1):
        xn = x - f(x) * (x - xp) / (f(x) - f(xp))
        if done(xn, x, eps): return xn, k
        xp, x = x, xn
    return x, N

def parabola(x0, x1, x2, eps=EPS, N=1000):
    a, b, c = x0, x1, x2                      # x_{n-2}, x_{n-1}, x_n
    for k in range(1, N + 1):
        f01 = (f(b) - f(a)) / (b - a)
        f12 = (f(c) - f(b)) / (c - b)
        f012 = (f12 - f01) / (c - a)
        w = f12 + (c - b) * f012              # = f[x_n,x_{n-1}] + f[x_n,x_{n-1},x_{n-2}](x_n-x_{n-1})
        disc = cmath.sqrt(w * w - 4 * f012 * f(c))
        den = w + disc if abs(w + disc) > abs(w - disc) else w - disc   # найменший за модулем крок
        xn = c - 2 * f(c) / den
        xn = xn.real if abs(xn.imag) < 1e-14 else xn
        if done(xn, c, eps): return xn, k
        a, b, c = b, c, xn
    return c, N

def inv_interp(x0, x1, x2, eps=EPS, N=1000):
    a, b, c = x0, x1, x2
    for k in range(1, N + 1):
        fa, fb, fc = f(a), f(b), f(c)
        # обернений многочлен Лагранжа при y=0 (3 вузли)
        xn = (a * fb * fc / ((fa - fb) * (fa - fc))
            + b * fa * fc / ((fb - fa) * (fb - fc))
            + c * fa * fb / ((fc - fa) * (fc - fb)))
        if done(xn, c, eps): return xn, k
        a, b, c = b, c, xn
    return c, N

# ---------- 6-7. Многочлени ----------
def read_coeffs(fname):
    with open(fname) as fi:
        return [float(t) for t in fi.read().split()]     # від старшого до вільного члена

def poly(a, x):
    s = 0
    for c in a: s = s * x + c
    return s

# ---------- 8. Ньютон + схема Горнера ----------
def newton_horner(a, x0, eps=EPS, N=1000):
    x = x0
    for k in range(1, N + 1):
        b = a[0]; c = a[0]                     # b -> P(x), c -> P'(x)
        for ai in a[1:-1]:
            b = ai + x * b
            c = b + x * c
        b = a[-1] + x * b
        xn = x - b / c
        if abs(xn - x) < eps and abs(poly(a, xn)) < eps: return xn, k
        x = xn
    return x, N

# ---------- 9. Метод Ліна (кубічне рівняння) ----------
def lin(a, p=1.0, q=1.0, eps=EPS, N=10000):
    a0, a1, a2, a3 = a                        # a0 = 1
    for k in range(1, N + 1):
        b0 = a0
        b1 = a1 - p * b0
        pn = (a2 - q * b0) / b1
        qn = a3 / b1
        if abs(pn - p) < eps and abs(qn - q) < eps:
            p, q = pn, qn
            break
        p, q = pn, qn
    d = cmath.sqrt(p * p - 4 * q)
    return (-p + d) / 2, (-p - d) / 2, p, q, k

# ---------- main ----------
if __name__ == "__main__":
    out = []
    P = lambda *s: (print(*s), out.append(" ".join(map(str, s))))
    P("Табуляція: f(x)=cos(x)-0.1x, [0;10], h=0.1 -> tab.txt")
    rts = tabulate()
    for x0, t in rts: P(f"  наближений корінь x0={x0:.4f} ({t})")
    for x0, t in rts:
        P(f"\n=== Корінь ({t}), x0={x0:.4f}, eps={EPS} ===")
        h = 0.1
        res = {
            "Проста ітерація (релаксація)": simple_iter(x0),
            "Ньютон":                       newton(x0),
            "Чебишев":                      chebyshev(x0),
            "Хорд":                         chord(x0, x0 + h),
            "Парабол":                      parabola(x0, x0 + h, x0 + 2*h),
            "Зворотна інтерполяція":        inv_interp(x0, x0 + h, x0 + 2*h),
        }
        P(f"{'Метод':32s}{'x*':>18s}{'ітерацій':>10s}{'|f(x*)|':>12s}")
        for name, (x, k) in res.items():
            P(f"{name:32s}{x:18.10f}{k:10d}{abs(f(x)):12.2e}")

    open("poly.txt", "w").write("1 -6 10 -8\n")
    a = read_coeffs("poly.txt")
    P("\n=== Алгебраїчне рівняння: x^3 - 6x^2 + 10x - 8 = 0 (poly.txt) ===")
    x, k = newton_horner(a, 5.0)
    P(f"Ньютон+Горнер: x = {x:.10f}, ітерацій = {k}, P(x) = {poly(a, x):.2e}")
    z1, z2, p, q, k = lin(a)
    P(f"Лін: p = {p:.10f}, q = {q:.10f}, ітерацій = {k}")
    P(f"  x2 = {z1:.8f}\n  x3 = {z2:.8f}")
    P(f"  перевірка |P(x2)| = {abs(poly(a, z1)):.2e}")
    open("results.txt", "w").write("\n".join(out))