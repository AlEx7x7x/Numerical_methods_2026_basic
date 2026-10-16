import random

N = 100         
SEED = 26       


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
    x_exact = [1.0] * N                                   # заданий розв'язок
    b = [sum(A[i][j] * x_exact[j] for j in range(N)) for i in range(N)]
    write_matrix("matrix_A.txt", A)
    write_vector("vector_b.txt", b)
    print(f"Згенеровано матрицю A ({N}x{N}) та вектор b, x_i = 1")


if __name__ == "__main__":
    main()
