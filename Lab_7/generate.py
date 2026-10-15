"""Лабораторна робота №7, програма 1: генерація СЛАР A*x = b
з діагональним переважанням."""
import random

N = 100          # розмірність системи
SEED = 27        # для відтворюваності результатів


def write_matrix(path, A):
    with open(path, "w") as f:
        f.write(f"{len(A)}\n")
        for row in A:
            f.write(" ".join(repr(v) for v in row) + "\n")


def write_vector(path, v):
    with open(path, "w") as f:
        f.write(f"{len(v)}\n")
        f.write("\n".join(repr(x) for x in v) + "\n")


def main():
    random.seed(SEED)
    A = [[random.uniform(-10.0, 10.0) for _ in range(N)] for _ in range(N)]
    for i in range(N):
        # діагональне переважання: a_ii > sum |a_ij|, j != i
        s = sum(abs(A[i][j]) for j in range(N) if j != i)
        A[i][i] = s + random.uniform(1.0, 10.0)
    x_exact = [1.0] * N                                   # заданий розв'язок
    b = [sum(A[i][j] * x_exact[j] for j in range(N)) for i in range(N)]
    write_matrix("matrix_A.txt", A)
    write_vector("vector_b.txt", b)
    print(f"Згенеровано матрицю A ({N}x{N}) з діагональним переважанням та вектор b, x_i = 1")


if __name__ == "__main__":
    main()
