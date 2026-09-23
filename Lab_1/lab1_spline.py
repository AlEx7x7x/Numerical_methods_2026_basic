import requests
import numpy as np
import matplotlib.pyplot as plt

url = ("https://api.open-elevation.com/api/v1/lookup?locations="
       "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
       "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
       "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
       "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
       "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
       "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
       "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106")

response = requests.get(url)
data = response.json()
results = data["results"]
n = len(results)
print("Кількість вузлів:", n)

print("\nТабуляція вузлів:")
print("№ | Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | "
          f"{point['longitude']:.6f} | {point['elevation']:.2f}")

def haversine(lat1, lon1, lat2, lon2):
    """Відстань між двома точками на сфері (у метрах)."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2)
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results], dtype=float)

distances = [0.0]
for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)
distances = np.array(distances)

print("\nТабуляція (відстань, висота):")
print("№ | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

plt.figure(figsize=(9, 5))
plt.plot(distances, elevations, "o-", color="tab:blue")
plt.xlabel("Кумулятивна відстань, м")
plt.ylabel("Висота, м")
plt.title("Профіль висоти маршруту Заросляк - Говерла (вихідні дані)")
plt.grid(True)
plt.tight_layout()
plt.savefig("profile_raw.png", dpi=150)
plt.close()

def natural_cubic_spline_coeffs(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x) - 1
    h = np.diff(x)

    alpha = np.zeros(n + 1)
    for i in range(1, n):
        alpha[i] = 3 * (y[i + 1] - y[i]) / h[i] - 3 * (y[i] - y[i - 1]) / h[i - 1]

    l = np.zeros(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)
    l[0] = 1.0

    for i in range(1, n):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[n] = 1.0
    z[n] = 0.0

    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)
    a = y[:n].copy()

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


a_c, b_c, c_c, d_c = natural_cubic_spline_coeffs(distances, elevations)

print("\nКоефіцієнти кубічного сплайна (повний набір вузлів):")
print(" i |      a      |      b      |      c      |      d")
for i in range(len(a_c)):
    print(f"{i:2d} | {a_c[i]:11.4f} | {b_c[i]:11.6f} | {c_c[i]:11.8f} | {d_c[i]:11.10f}")

xx_full = np.linspace(distances[0], distances[-1], 500)
yy_full = spline_eval(distances, a_c, b_c, c_c, d_c, xx_full)

def subsample(n_target, distances, elevations):
    idx = np.unique(np.linspace(0, len(distances) - 1, n_target).astype(int))
    return distances[idx], elevations[idx]

plt.figure(figsize=(10, 6))
plt.plot(distances, elevations, "ko", label="Вихідні точки (усі)")

colors = {10: "tab:orange", 15: "tab:green", 20: "tab:red"}
splines_by_n = {}

for n_nodes in (10, 15, 20):
    xs, ys = subsample(n_nodes, distances, elevations)
    ac, bc, cc, dc = natural_cubic_spline_coeffs(xs, ys)
    xx = np.linspace(xs[0], xs[-1], 500)
    yy = spline_eval(xs, ac, bc, cc, dc, xx)
    splines_by_n[n_nodes] = (xs, ys, xx, yy)
    plt.plot(xx, yy, color=colors[n_nodes], label=f"Сплайн, {n_nodes} вузлів")

plt.xlabel("Кумулятивна відстань, м")
plt.ylabel("Висота, м")
plt.title("Кубічні сплайни профілю висоти для різної кількості вузлів")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("profile_splines_10_15_20.png", dpi=150)
plt.close()

print("\nОцінка точності для різної кількості вузлів:")
for n_nodes in (10, 15, 20):
    xs, ys, xx, yy = splines_by_n[n_nodes]
    y_ref = spline_eval(distances, a_c, b_c, c_c, d_c, xx)
    err = np.abs(yy - y_ref)
    print(f"  {n_nodes} вузлів: max похибка = {np.max(err):.3f} м, "
          f"середня похибка = {np.mean(err):.3f} м")

xs, ys, xx, yy = splines_by_n[10]
y_ref = spline_eval(distances, a_c, b_c, c_c, d_c, xx)
error = yy - y_ref

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
axes[0].plot(xx, y_ref, label="Еталон (повний набір вузлів)", color="black")
axes[0].plot(xx, yy, "--", label="Наближення (10 вузлів)", color="tab:orange")
axes[0].set_ylabel("Висота, м")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(xx, error, color="tab:red")
axes[1].set_xlabel("Кумулятивна відстань, м")
axes[1].set_ylabel("Похибка, м")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("profile_error.png", dpi=150)
plt.close()

print("\n--- Характеристики маршруту ---")
print("Загальна довжина маршруту (м):", distances[-1])

total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))
print("Сумарний набір висоти (м):", total_ascent)
print("Сумарний спуск (м):", total_descent)

grad_full = np.gradient(yy_full, xx_full) * 100
print("Максимальний підйом (%):", np.max(grad_full))
print("Максимальний спуск (%):", np.min(grad_full))
print("Середній градієнт (%):", np.mean(np.abs(grad_full)))

mass = 80
g = 9.81
energy = mass * g * total_ascent
print("Механічна робота (Дж):", energy)
print("Механічна робота (кДж):", energy / 1000)
print("Енергія (ккал):", energy / 4184)

print("\nГотово. Збережено графіки: profile_raw.png, "
      "profile_splines_10_15_20.png, profile_error.png")