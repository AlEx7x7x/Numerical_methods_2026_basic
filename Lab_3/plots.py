import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_variance(degrees, variances, path):
    plt.figure(figsize=(7, 4))
    plt.plot(degrees, variances, "o-")
    plt.xlabel("Степінь многочлена m")
    plt.ylabel("Дисперсія σ²")
    plt.title("Залежність дисперсії від степеня")
    plt.xticks(degrees)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_approximation(x, y, xs_dense, ys_dense, x_future, y_future, m, path):
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, "ko", label="Фактичні дані")
    plt.plot(xs_dense, ys_dense, "r-", label=f"Многочлен m={m}")
    plt.plot(x_future, y_future, "gs", label="Прогноз")
    plt.xlabel("Місяць")
    plt.ylabel("Температура, °C")
    plt.title("Апроксимація методом найменших квадратів")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_errors(x, errors_by_m, path):
    plt.figure(figsize=(9, 5))
    for m, err in errors_by_m.items():
        plt.plot(x, err, "o-", markersize=3, label=f"m={m}")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("Місяць")
    plt.ylabel("ε(x) = y − P(x)")
    plt.title("Похибка апроксимації")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()