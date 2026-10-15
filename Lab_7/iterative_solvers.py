"""Лабораторна робота №7, програма 2: ітераційні методи розв'язку СЛАР
(проста ітерація, Якобі, Зейдель)."""
import math

EPS = 1e-10          # задана точність
MAX_ITER = 100000    # максимальна кількість ітерацій


# ---------- введення / виведення ----------
def read_matrix(path):
    with open(path) as f:
        n = int(f.readline())
        return [[float(v) for v in f.readline().split()] for _ in range(n)]


def read_vector(path):
    with open(path) as f:
        n = int(f.readline())
        return [float(f.readline()) for _ in range(n)]


# ---------- допоміжні операції ----------
def mat_vec(A, x):
    return [sum(a * xj for a, xj in zip(row, x)) for row in A]


def vec_norm(v):
    """Евклідова норма вектора."""
    return math.sqrt(sum(t * t for t in v))


def mat_norm(A):
    """Норма матриці ||A||_inf = max_i sum_j |a_ij| (максимальна сума по рядках)."""
    return max(sum(abs(a) for a in row) for row in A)


def diff(a, b):
    return [p - q for p, q in zip(a, b)]


def residual_norm(A, x, b):
    return vec_norm(diff(mat_vec(A, x), b))


def is_diag_dominant(A):
    return all(abs(A[i][i]) > sum(abs(A[i][j]) for j in range(len(A)) if j != i)
               for i in range(len(A)))


def simple_iter_tau(A):
    """Параметр tau: ||E - tau*A||_inf < 1 для матриці з діагональним
    переважанням і додатною діагоналлю при tau = 1 / max a_ii."""
    return 1.0 / max(A[i][i] for i in range(len(A)))


def simple_iter_matrix_norm(A, tau):
    """||E - tau*A||_inf."""
    n = len(A)
    B = [[(1.0 if i == j else 0.0) - tau * A[i][j] for j in range(n)] for i in range(n)]
    return mat_norm(B)


def jacobi_matrix_norm(A):
    """||B||_inf для B = -D^{-1}(L+U): max_i sum_{j!=i} |a_ij| / |a_ii|."""
    n = len(A)
    return max(sum(abs(A[i][j]) for j in range(n) if j != i) / abs(A[i][i])
               for i in range(n))


# ---------- ітераційні методи ----------
# Умова завершення: ||x(k+1) - x(k)|| < eps  та  ||A x(k+1) - b|| < eps.
# Повертають (розв'язок, історія [(k, ||dx||, ||r||, x), ...]).
def _finish(A, b, x_new, x_old, eps):
    dx = vec_norm(diff(x_new, x_old))
    r = residual_norm(A, x_new, b)
    return dx, r, (dx < eps and r < eps)


def simple_iteration(A, b, x0, tau, eps=EPS, max_iter=MAX_ITER, callback=None):
    """x(k+1) = x(k) - tau*(A x(k) - b)."""
    x = x0[:]
    n = len(b)
    for k in range(1, max_iter + 1):
        Ax = mat_vec(A, x)
        x_new = [x[i] - tau * (Ax[i] - b[i]) for i in range(n)]
        dx, r, ok = _finish(A, b, x_new, x, eps)
        if callback:
            callback(k, dx, r, x_new)
        x = x_new
        if ok:
            return x, k
    return x, max_iter


def jacobi(A, b, x0, eps=EPS, max_iter=MAX_ITER, callback=None):
    """x_i(k+1) = (b_i - sum_{j!=i} a_ij x_j(k)) / a_ii."""
    x = x0[:]
    n = len(b)
    for k in range(1, max_iter + 1):
        x_new = [(b[i] - sum(A[i][j] * x[j] for j in range(n) if j != i)) / A[i][i]
                 for i in range(n)]
        dx, r, ok = _finish(A, b, x_new, x, eps)
        if callback:
            callback(k, dx, r, x_new)
        x = x_new
        if ok:
            return x, k
    return x, max_iter


def seidel(A, b, x0, eps=EPS, max_iter=MAX_ITER, callback=None):
    """x_i(k+1) = (b_i - sum_{j<i} a_ij x_j(k+1) - sum_{j>i} a_ij x_j(k)) / a_ii."""
    x = x0[:]
    n = len(b)
    for k in range(1, max_iter + 1):
        x_new = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        dx, r, ok = _finish(A, b, x_new, x, eps)
        if callback:
            callback(k, dx, r, x_new)
        x = x_new
        if ok:
            return x, k
    return x, max_iter


def main():
    A = read_matrix("matrix_A.txt")
    b = read_vector("vector_b.txt")
    n = len(A)
    exact = [1.0] * n
    print(f"n = {n}, діагональне переважання: {is_diag_dominant(A)}, eps = {EPS:g}")

    x0 = [0.0] * n                       # початкове наближення
    tau = simple_iter_tau(A)
    print(f"||A||_inf = {mat_norm(A):.4f}, tau = {tau:.6f}")
    print(f"||E - tau*A||_inf = {simple_iter_matrix_norm(A, tau):.4f}, "
          f"||B_Якобі||_inf = {jacobi_matrix_norm(A):.4f}")

    methods = (("Проста ітерація", lambda: simple_iteration(A, b, x0, tau)),
               ("Якобі", lambda: jacobi(A, b, x0)),
               ("Зейдель", lambda: seidel(A, b, x0)))
    for name, run in methods:
        x, k = run()
        err = vec_norm(diff(x, exact))
        print(f"{name:16s}: ітерацій = {k:4d}, ||x-x*|| = {err:.3e}, "
              f"||Ax-b|| = {residual_norm(A, x, b):.3e}")
        fname = {"Проста ітерація": "solution_simple.txt", "Якобі": "solution_jacobi.txt",
                 "Зейдель": "solution_seidel.txt"}[name]
        with open(fname, "w") as f:
            f.write("\n".join(repr(v) for v in x) + "\n")


if __name__ == "__main__":
    main()
