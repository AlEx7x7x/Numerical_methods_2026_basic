"""Лабораторна робота №6, програма 2: LU-розклад та ітераційне уточнення.
L - нижня трикутна, U - верхня трикутна з одиничною діагоналлю (A = L*U)."""
import math

EPS = 1e-12      # задана точність


# ---------- введення / виведення ----------
def read_matrix(path):
    with open(path) as f:
        n = int(f.readline())
        return [[float(v) for v in f.readline().split()] for _ in range(n)]


def read_vector(path):
    with open(path) as f:
        n = int(f.readline())
        return [float(f.readline()) for _ in range(n)]


def write_lu(path, L, U):
    with open(path, "w") as f:
        for name, M in (("L", L), ("U", U)):
            f.write(f"{name}\n")
            for row in M:
                f.write(" ".join(f"{v:.10e}" for v in row) + "\n")


# ---------- допоміжні операції ----------
def mat_vec(A, x):
    return [sum(a * xj for a, xj in zip(row, x)) for row in A]


def norm(v):
    """Евклідова норма вектора."""
    return math.sqrt(sum(t * t for t in v))


# ---------- LU-розклад ----------
def lu_decompose(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        U[i][i] = 1.0
    for k in range(n):
        for i in range(k, n):                    # k-й стовпець L
            L[i][k] = A[i][k] - sum(L[i][s] * U[s][k] for s in range(k))
        if abs(L[k][k]) < 1e-300:
            raise ZeroDivisionError("нульовий діагональний елемент L")
        for j in range(k + 1, n):                # k-й рядок U
            U[k][j] = (A[k][j] - sum(L[k][s] * U[s][j] for s in range(k))) / L[k][k]
    return L, U


def lu_solve(L, U, b):
    n = len(b)
    y = [0.0] * n
    for i in range(n):                           # L*y = b
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):               # U*x = y
        x[i] = y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))
    return x


# ---------- ітераційне уточнення ----------
def refine(A, L, U, b, x0, eps=EPS, max_iter=50):
    x = x0[:]
    history = []
    for it in range(1, max_iter + 1):
        Ax = mat_vec(A, x)
        r = [bi - ai for bi, ai in zip(b, Ax)]          # нев'язка
        dx = lu_solve(L, U, r)                          # A*dx = r
        x = [xi + di for xi, di in zip(x, dx)]
        rel_dx = norm(dx) / norm(x)
        res = norm([bi - ai for bi, ai in zip(b, mat_vec(A, x))])
        history.append((it, norm(dx), res))
        if norm(dx) < eps and res < eps * max(1.0, norm(b)):
            break
    return x, history


def main():
    A = read_matrix("matrix_A.txt")
    b = read_vector("vector_b.txt")
    n = len(A)

    L, U = lu_decompose(A)
    write_lu("LU_decomposition.txt", L, U)
    print(f"n = {n}; LU-розклад записано у LU_decomposition.txt")

    x0 = lu_solve(L, U, b)
    exact = [1.0] * n
    err0 = norm([a - e for a, e in zip(x0, exact)])
    res0 = norm([bi - ai for bi, ai in zip(b, mat_vec(A, x0))])
    print(f"Початковий розв'язок: ||x-x*|| = {err0:.3e}, ||b-Ax|| = {res0:.3e}")

    x, hist = refine(A, L, U, b, x0)
    for it, dxn, res in hist:
        print(f"  ітерація {it}: ||dx|| = {dxn:.3e}, ||b-Ax|| = {res:.3e}")
    err = norm([a - e for a, e in zip(x, exact)])
    print(f"Уточнений розв'язок: ||x-x*|| = {err:.3e}, ітерацій: {len(hist)}")
    with open("solution.txt", "w") as f:
        f.write("\n".join(repr(v) for v in x) + "\n")


if __name__ == "__main__":
    main()
