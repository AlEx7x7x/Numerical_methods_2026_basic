"""Лабораторна робота №4
Чисельне диференціювання, метод Рунге-Ромберга, метод Ейткена.
M(t) = 50*exp(-0.1 t) + 5*sin(t),  x0 = 1
"""
import sys, os, math
import numpy as np
import matplotlib.pyplot as plt

# --- коректний вивід українських літер у терміналі ---
if os.name == "nt":
    os.system("chcp 65001 > nul")          # кодова сторінка UTF-8 для Windows
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
plt.rcParams["font.family"] = "DejaVu Sans"  # підтримує кирилицю

# ---------- 1. Функція, похідна, точне значення ----------
f  = lambda t: 50 * math.exp(-0.1 * t) + 5 * math.sin(t)
df = lambda t: -5 * math.exp(-0.1 * t) + 5 * math.cos(t)   # явний вираз похідної
x0 = 1.0
exact = df(x0)
print(f"1. Точне значення M'({x0}) = {exact:.12f}\n")

# ---------- центральна різниця ----------
def D(h):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

# ---------- 2. Залежність похибки від кроку h = 1e-20 ... 1e3 ----------
print("2. Залежність похибки від кроку")
print(f"{'h':>10} {'y0prime(h)':>22} {'R(h)':>12}")
for k in range(-20, 4):
    h = 10.0 ** k
    print(f"{h:10.0e} {D(h):22.12f} {abs(D(h) - exact):12.3e}")

# Пошук оптимального кроку (дрібне сканування за степенем 10)
exps = np.arange(-20, 3.0001, 0.01)
errs = [abs(D(10.0 ** e) - exact) for e in exps]
i = int(np.argmin(errs))
h0, R0 = 10.0 ** exps[i], errs[i]
print(f"\nОптимальний крок h0 = {h0:.1e}  (формат {h0:.0e}),  R0 = {R0:.3e}")
print("(Мінімум частково випадковий через округлення; стійкий оптимум h ~ 1e-5)\n")

# ---------- 3-5. Крок h = 1e-3, значення при h та 2h, похибка R1 ----------
h = 1e-3
y1 = D(h)          # y0'(h)
y2 = D(2 * h)      # y0'(2h)
R1 = abs(y1 - exact)
print(f"3-5. h = {h}")
print(f"  y0'(h)  = {y1:.12f}")
print(f"  y0'(2h) = {y2:.12f}")
print(f"  R1 = |y0'(h) - y'(x0)| = {R1:.3e}\n")

# ---------- 6. Метод Рунге-Ромберга ----------
yR = y1 + (y1 - y2) / 3
R2 = abs(yR - exact)
print("6. Метод Рунге-Ромберга")
print(f"  y'_R = {yR:.12f}")
print(f"  R2 = {R2:.3e}   (R1/R2 = {R1 / R2:.2e} разів менша похибка)\n")

# ---------- 7. Метод Ейткена (кроки h, 2h, 4h) ----------
y4 = D(4 * h)      # y0'(4h)
yE = (y2 ** 2 - y4 * y1) / (2 * y2 - (y4 + y1))
p  = math.log(abs((y4 - y2) / (y2 - y1))) / math.log(2)
R3 = abs(yE - exact)
print("7. Метод Ейткена")
print(f"  y0'(4h) = {y4:.12f}")
print(f"  y'_E = {yE:.12f}")
print(f"  порядок точності p = {p:.6f}")
print(f"  R3 = {R3:.3e}\n")

# ---------- Порівняльна таблиця ----------
print("Порівняння")
print(f"{'Метод':<28}{'Значення':>20}{'Похибка':>14}")
for name, v in [("Точне", exact),
                (f"Центральна різниця h={h:g}", y1),
                (f"Центральна різниця 2h={2*h:g}", y2),
                (f"Центральна різниця h0={h0:.1e}", D(h0)),
                ("Рунге-Ромберг", yR),
                ("Ейткен", yE)]:
    print(f"{name:<28}{v:20.12f}{abs(v - exact):14.3e}")

# ---------- 8. Графіки ----------
hs = 10.0 ** exps

# Графік 1: функція M(t) та її похідна
t = np.linspace(0, 20, 500)
fig1, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t, [f(v) for v in t]); ax[0].plot(x0, f(x0), "ro")
ax[0].set_title("Вологість ґрунту M(t)"); ax[0].set_xlabel("t"); ax[0].set_ylabel("M(t)")
ax[1].plot(t, [df(v) for v in t]); ax[1].plot(x0, exact, "ro", label=f"M'({x0:g}) = {exact:.4f}")
ax[1].set_title("Швидкість зміни M'(t)"); ax[1].set_xlabel("t"); ax[1].set_ylabel("M'(t)"); ax[1].legend()
for a in ax: a.grid(alpha=.3)
fig1.tight_layout(); fig1.savefig("graph1_function.png", dpi=150)

# Графік 2: залежність похибки від кроку (п.2)
fig2 = plt.figure(figsize=(8, 5))
plt.loglog(hs, np.maximum(errs, 1e-17), lw=1, label="R(h)")
mask = hs > 1e-4
plt.loglog(hs[mask], 0.46 * hs[mask] ** 2, "g--", lw=1, label="~ h² (похибка методу)")
mask2 = hs < 1e-4
plt.loglog(hs[mask2], 1e-16 * 50 / hs[mask2], "m--", lw=1, label="~ 1/h (похибка округлення)")
plt.axvline(h0, color="r", ls="--", label=f"h0 = {h0:.1e}")
plt.ylim(1e-14, 1e6)
plt.xlabel("h"); plt.ylabel("R(h)"); plt.title("Залежність похибки від кроку h")
plt.grid(True, which="both", alpha=.3); plt.legend()
fig2.savefig("graph2_error_vs_h.png", dpi=150, bbox_inches="tight")

# Графік 3: похибка трьох методів залежно від h
hh = 10.0 ** np.arange(-1, -5.01, -0.25)
eD, eR, eE = [], [], []
for hv in hh:
    a, b, c = D(hv), D(2 * hv), D(4 * hv)
    rr = a + (a - b) / 3
    den = 2 * b - (c + a)
    ae = (b * b - c * a) / den if den != 0 else a
    eD.append(abs(a - exact)); eR.append(abs(rr - exact)); eE.append(abs(ae - exact))
fig3 = plt.figure(figsize=(8, 5))
plt.loglog(hh, np.maximum(eD, 1e-17), "o-", label="Центральна різниця")
plt.loglog(hh, np.maximum(eR, 1e-17), "s-", label="Рунге–Ромберг")
plt.loglog(hh, np.maximum(eE, 1e-17), "^-", label="Ейткен")
plt.axvline(h, color="gray", ls=":", label=f"h = {h:g}")
plt.xlabel("h"); plt.ylabel("Абсолютна похибка")
plt.title("Порівняння точності методів"); plt.grid(True, which="both", alpha=.3); plt.legend()
fig3.savefig("graph3_methods_vs_h.png", dpi=150, bbox_inches="tight")

# Графік 4: стовпчикова діаграма похибок при h = 1e-3
names = ["h", "2h", "h0", "Рунге–\nРомберг", "Ейткен"]
vals = [R1, abs(y2 - exact), abs(D(h0) - exact), R2, R3]
fig4 = plt.figure(figsize=(8, 5))
bars = plt.bar(names, vals, color=["#4c72b0", "#4c72b0", "#55a868", "#dd8452", "#c44e52"])
plt.yscale("log"); plt.ylabel("Абсолютна похибка")
plt.title(f"Похибки при h = {h:g}")
for b_, v_ in zip(bars, vals):
    plt.text(b_.get_x() + b_.get_width() / 2, v_, f"{v_:.1e}", ha="center", va="bottom")
plt.grid(True, axis="y", which="both", alpha=.3)
fig4.savefig("graph4_comparison.png", dpi=150, bbox_inches="tight")

print("\nГрафіки збережено: graph1_function.png, graph2_error_vs_h.png, "
      "graph3_methods_vs_h.png, graph4_comparison.png")
plt.show()