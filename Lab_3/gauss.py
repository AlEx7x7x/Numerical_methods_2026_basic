def gauss_solve(A, b):
    n = len(b)
    A = [row[:] for row in A]   
    b = b[:]

    for k in range(n):
        max_row = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[max_row][k]) < 1e-14:
            raise ValueError("Матриця вироджена")
        if max_row != k:
            A[k], A[max_row] = A[max_row], A[k]
            b[k], b[max_row] = b[max_row], b[k]
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * sol[j] for j in range(i + 1, n))
        sol[i] = (b[i] - s) / A[i][i]
    return sol