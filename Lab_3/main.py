from data_io import read_csv
from lsq import fit, polynomial, errors, variance
from plots import plot_variance, plot_approximation, plot_errors

MAX_DEGREE = 4
FUTURE_MONTHS = [25.0, 26.0, 27.0]


def main():
    x, y = read_csv("data.csv")

    coefs, variances, errs = {}, [], {}
    degrees = list(range(1, MAX_DEGREE + 1))
    for m in degrees:
        coefs[m] = fit(x, y, m)
        v = variance(x, y, coefs[m])
        variances.append(v)
        errs[m] = errors(x, y, coefs[m])
        print(f"m={m}: σ² = {v:.4f}")
        print("   коефіцієнти:", [round(c, 6) for c in coefs[m]])

    opt_m = degrees[variances.index(min(variances))]
    print(f"\nОптимальний степінь: m = {opt_m}")

    coef = coefs[opt_m]
    y_future = polynomial(FUTURE_MONTHS, coef)
    print("\nПрогноз температури:")
    for xm, ym in zip(FUTURE_MONTHS, y_future):
        print(f"  місяць {int(xm)}: {ym:.2f} °C")

    xs = [1 + i * 0.1 for i in range(int((max(FUTURE_MONTHS) - 1) / 0.1) + 1)]
    plot_variance(degrees, variances, "variance.png")
    plot_approximation(x, y, xs, polynomial(xs, coef),
                       FUTURE_MONTHS, y_future, opt_m, "approximation.png")
    plot_errors(x, errs, "errors.png")
    print("\nГрафіки: variance.png, approximation.png, errors.png")


if __name__ == "__main__":
    main()