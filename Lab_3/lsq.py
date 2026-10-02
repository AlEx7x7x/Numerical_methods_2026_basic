from gauss import gauss_solve


def form_matrix(x, m):
    return [[sum(xk ** (i + j) for xk in x) for j in range(m + 1)]
            for i in range(m + 1)]


def form_vector(x, y, m):
    return [sum(yk * xk ** i for xk, yk in zip(x, y)) for i in range(m + 1)]


def fit(x, y, m):
    return gauss_solve(form_matrix(x, m), form_vector(x, y, m))


def polynomial(x, coef):
    def p(t):
        r = 0.0
        for c in reversed(coef):
            r = r * t + c
        return r
    if isinstance(x, (list, tuple)):
        return [p(t) for t in x]
    return p(x)


def errors(x, y, coef):
    return [yk - pk for yk, pk in zip(y, polynomial(list(x), coef))]


def variance(x, y, coef):
    n, m = len(x), len(coef) - 1
    return sum(e * e for e in errors(x, y, coef)) / (n - m - 1)