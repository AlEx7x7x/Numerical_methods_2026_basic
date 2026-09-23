# -*- coding: utf-8 -*-
"""
Лабораторна робота №2. Розділені різниці. Інтерполяційні многочлени
Ньютона та факторіальних многочленів.
Варіант 5. Оптимізація ігрового рушія. Прогнозування FPS(objects).
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# Примусово вмикаємо UTF-8 для виводу в консоль (виправляє "кракозябри"
# з українським текстом у стандартній консолі Windows, яка за замовчуванням
# використовує кодування cp1251)
sys.stdout.reconfigure(encoding="utf-8")

# =========================================================
# 1. Зчитування даних з CSV
# =========================================================
def read_data(filename):
    x, y = [], []
    with open(filename, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row["objects"]))
            y.append(float(row["fps"]))
    return np.array(x), np.array(y)


# =========================================================
# 2. Таблиця розділених різниць та поліном Ньютона
#    (працює для ДОВІЛЬНОЇ, у тому числі нерівномірної, сітки)
# =========================================================
def divided_diff_table(x, y):
    """
    coef[i][j] = розділена різниця f[x_i, x_{i+1}, ..., x_{i+j}] j-го порядку.
    Верхній рядок coef[0] містить коефіцієнти полінома Ньютона:
    c0 = f[x0], c1 = f[x0,x1], c2 = f[x0,x1,x2], ...
    """
    n = len(x)
    coef = np.zeros((n, n))
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef


def newton_poly_eval(x_data, coef_row0, x_query):
    """
    Обчислення значення полінома Ньютона за схемою Горнера:
    P(x) = c0 + (x-x0)*(c1 + (x-x1)*(c2 + ... ))
    """
    n = len(x_data)
    x_query = np.atleast_1d(np.asarray(x_query, dtype=float))
    p = np.full_like(x_query, coef_row0[n - 1])
    for k in range(1, n):
        p = coef_row0[n - 1 - k] + (x_query - x_data[n - 1 - k]) * p
    return p


# =========================================================
# 3. Факторіальні многочлени (потребують РІВНОВІДДАЛЕНИХ вузлів)
#    Реальні вузли варіанту 5 (100,200,400,800,1600) НЕ рівновіддалені,
#    але рівновіддалені у логарифмічній шкалі: u = log2(objects/objects0).
#    Це дозволяє коректно застосувати факторіальні многочлени.
# =========================================================
def forward_diff_table(y):
    """Таблиця скінченних різниць вперед: table[i][j] = Δ^j y_i."""
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]
    return table


def factorial_poly_eval(diffs_row0, u, n_nodes):
    """
    P(u) = y0 + Δy0 * u^(1) + (Δ^2 y0 / 2!) * u^(2) + ...
    де u^(k) = u*(u-1)*...*(u-k+1) -- факторіальний многочлен (спадний факторіал).
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    p = np.full_like(u, diffs_row0[0])
    term = np.ones_like(u)
    for k in range(1, n_nodes):
        term = term * (u - (k - 1))
        p = p + diffs_row0[k] / factorial(k) * term
    return p


# =========================================================
# 4. Основні дані варіанту (Objects -> FPS)
# =========================================================
objects = np.array([100, 200, 400, 800, 1600], dtype=float)
fps = np.array([120, 110, 90, 65, 40], dtype=float)

# Збережемо у CSV, як вимагає методичка
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["objects", "fps"])
    for o, v in zip(objects, fps):
        writer.writerow([int(o), v])

objects, fps = read_data("data.csv")
print("Вхідні дані:")
for o, v in zip(objects, fps):
    print(f"  Objects={o:6.0f}  FPS={v:6.1f}")

# =========================================================
# 5. Таблиця розділених різниць (метод Ньютона, нерівномірна сітка)
# =========================================================
coef = divided_diff_table(objects, fps)
print("\nТаблиця розділених різниць (рядок 0 = коефіцієнти полінома Ньютона):")
n = len(objects)
for j in range(n):
    row = [f"{coef[i][j]: .6f}" for i in range(n - j)]
    print(f"  порядок {j}: {row}")

# =========================================================
# 6. Прогноз для n = 1000 об'єктів -- метод Ньютона (загальний)
# =========================================================
x_target = 1000.0
fps_newton = newton_poly_eval(objects, coef[0], x_target)[0]
print(f"\nПрогноз FPS при {x_target:.0f} об'єктах (Ньютон, розділені різниці): {fps_newton:.3f}")

# =========================================================
# 7. Прогноз для n = 1000 -- факторіальні многочлени
#    (через заміну змінних u = log2(objects/objects[0]))
# =========================================================
u_nodes = np.log2(objects / objects[0])          # 0, 1, 2, 3, 4 -- рівновіддалені, крок 1
diffs = forward_diff_table(fps)
u_target = np.log2(x_target / objects[0])
fps_factorial = factorial_poly_eval(diffs[0], u_target, n)[0]
print(f"Прогноз FPS при {x_target:.0f} об'єктах (факторіальні многочлени, змінна u): {fps_factorial:.3f}")

# ВАЖЛИВО: це поліном від u = log2(objects/objects0), а не від objects
# напряму, тому він не зобов'язаний числово збігатися з поліномом Ньютона
# від objects (це дві різні за формою моделі, що просто проходять через
# ті самі 5 точок). Щоб перевірити коректність реалізації факторіальних
# многочленів, порівняємо їх з методом Ньютона, побудованим У ТІЙ САМІЙ
# змінній u -- результати мають збігтися практично точно:
coef_u = divided_diff_table(u_nodes, fps)
fps_newton_u = newton_poly_eval(u_nodes, coef_u[0], u_target)[0]
print(f"Перевірка: метод Ньютона у змінній u дає {fps_newton_u:.6f}, "
      f"факторіальні многочлени дають {fps_factorial:.6f} "
      f"(різниця {abs(fps_newton_u - fps_factorial):.2e} -- підтверджує коректність реалізації)")
print(f"Різниця між прогнозом у змінній x та у змінній u: "
      f"{abs(fps_newton - fps_factorial):.3f} -- це дві РІЗНІ моделі "
      f"(поліном від objects і поліном від log2(objects)), тому їх прогнози "
      f"на нерівномірній сітці природно відрізняються.")

# =========================================================
# 8. Табуляція функції, полінома Ньютона та похибки на відрізку
# =========================================================
xx = np.linspace(objects[0], objects[-1], 400)
yy_newton = newton_poly_eval(objects, coef[0], xx)

plt.figure(figsize=(9, 5))
plt.plot(objects, fps, "ko", markersize=8, label="Експериментальні точки")
plt.plot(xx, yy_newton, "-", color="tab:blue", label="Поліном Ньютона (degree 4)")
plt.axvline(x_target, color="gray", linestyle=":", label=f"n={x_target:.0f}")
plt.plot([x_target], [fps_newton], "r*", markersize=14, label=f"Прогноз: {fps_newton:.1f} FPS")
plt.xlabel("Кількість об'єктів (Objects)")
plt.ylabel("FPS")
plt.title("Інтерполяція FPS(Objects) поліномом Ньютона")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fps_newton.png", dpi=150)
plt.close()

# =========================================================
# 9. Дослідницька частина: вплив кількості вузлів (5, 10, 20)
#    Оскільки реальних експериментальних точок лише 5, для
#    дослідження впливу КІЛЬКОСТІ ВУЗЛІВ на 10 і 20 точках
#    використовується згенерована еталонна крива -- натуральний
#    кубічний сплайн через 5 реальних точок (як у лаб. роботі №1).
#    Це дозволяє коректно продемонструвати ефект Рунге, який
#    неможливо показати лише на 5 реальних вимірюваннях.
# =========================================================
def natural_cubic_spline_coeffs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x) - 1
    h = np.diff(x)
    alpha = np.zeros(n + 1)
    for i in range(1, n):
        alpha[i] = 3 * (y[i + 1] - y[i]) / h[i] - 3 * (y[i] - y[i - 1]) / h[i - 1]
    l = np.zeros(n + 1); mu = np.zeros(n + 1); z = np.zeros(n + 1)
    l[0] = 1.0
    for i in range(1, n):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    l[n] = 1.0
    z[n] = 0.0
    c = np.zeros(n + 1); b = np.zeros(n); d = np.zeros(n); a = y[:n].copy()
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])
    return a, b, c[:n], d


def spline_eval(x_nodes, a, b, c, d, x_query):
    x_nodes = np.asarray(x_nodes)
    x_query = np.atleast_1d(np.asarray(x_query, dtype=float))
    result = np.zeros_like(x_query)
    n_seg = len(a)
    for k, xq in enumerate(x_query):
        i = np.searchsorted(x_nodes, xq) - 1
        i = min(max(i, 0), n_seg - 1)
        dx = xq - x_nodes[i]
        result[k] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    return result


a_s, b_s, c_s, d_s = natural_cubic_spline_coeffs(objects, fps)
xx_ref = np.linspace(objects[0], objects[-1], 400)
yy_ref = spline_eval(objects, a_s, b_s, c_s, d_s, xx_ref)

plt.figure(figsize=(10, 6))
plt.plot(xx_ref, yy_ref, "k--", linewidth=1.5, label="Еталонна крива (сплайн через реальні точки)")
plt.plot(objects, fps, "ko", markersize=8, label="Реальні експериментальні точки")

colors = {5: "tab:blue", 10: "tab:orange", 20: "tab:red"}
poly_by_n = {}
for n_nodes in (5, 10, 20):
    x_sample = np.linspace(objects[0], objects[-1], n_nodes)
    y_sample = spline_eval(objects, a_s, b_s, c_s, d_s, x_sample)
    coef_n = divided_diff_table(x_sample, y_sample)
    xx_n = np.linspace(objects[0], objects[-1], 400)
    yy_n = newton_poly_eval(x_sample, coef_n[0], xx_n)
    poly_by_n[n_nodes] = (x_sample, y_sample, xx_n, yy_n)
    plt.plot(xx_n, yy_n, color=colors[n_nodes], label=f"Ньютон, {n_nodes} вузлів (degree {n_nodes-1})")

plt.xlabel("Кількість об'єктів (Objects)")
plt.ylabel("FPS")
plt.title("Вплив кількості вузлів на поліном Ньютона (ефект Рунге)")
plt.legend()
plt.grid(True)
plt.ylim(min(fps) - 60, max(fps) + 60)  # обмежуємо вісь, щоб побачити криву при осциляціях
plt.tight_layout()
plt.savefig("fps_runge_effect.png", dpi=150)
plt.close()

# =========================================================
# 10. Похибка інтерполяції для 5 / 10 / 20 вузлів
# =========================================================
print("\nОцінка точності для різної кількості вузлів (відносно еталонної кривої):")
for n_nodes in (5, 10, 20):
    x_sample, y_sample, xx_n, yy_n = poly_by_n[n_nodes]
    y_ref_n = spline_eval(objects, a_s, b_s, c_s, d_s, xx_n)
    err = np.abs(yy_n - y_ref_n)
    print(f"  {n_nodes:2d} вузлів: max похибка = {np.max(err):10.2f}  "
          f"середня похибка = {np.mean(err):8.2f}")

plt.figure(figsize=(9, 5))
for n_nodes in (5, 10, 20):
    x_sample, y_sample, xx_n, yy_n = poly_by_n[n_nodes]
    y_ref_n = spline_eval(objects, a_s, b_s, c_s, d_s, xx_n)
    err = yy_n - y_ref_n
    plt.plot(xx_n, err, color=colors[n_nodes], label=f"{n_nodes} вузлів")
plt.xlabel("Кількість об'єктів (Objects)")
plt.ylabel("Похибка (Ньютон - еталон)")
plt.title("Похибка полінома Ньютона для різної кількості вузлів")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fps_error_by_nodes.png", dpi=150)
plt.close()

# =========================================================
# 11. Мінімальна кількість об'єктів, при якій FPS >= 60
#     (розв'язується чисельно по поліному Ньютона degree 4)
# =========================================================
xx_dense = np.linspace(objects[0], objects[-1], 5000)
yy_dense = newton_poly_eval(objects, coef[0], xx_dense)
below_60 = xx_dense[yy_dense <= 60]
if len(below_60) > 0:
    n_critical = below_60[0]
    print(f"\nFPS опускається до 60 приблизно при {n_critical:.0f} об'єктах "
          f"(за поліномом Ньютона на реальних 5 точках)")
else:
    print("\nУ межах [100, 1600] FPS не опускається до 60 (за побудованим поліномом)")

print("\nГотово. Збережено: data.csv, fps_newton.png, "
      "fps_runge_effect.png, fps_error_by_nodes.png")