import numpy as np
import random


def main():
    adj = []
    matrix = []
    with open('adj_matrix.txt') as file:
        for line in file:
            x = [int(i) for i in line.strip().split(' ')]
            matrix.append(x)
            adj.append(list(x))
    size = adj[0][0]
    y_true = page_rank_vector(matrix, adj, size)
    print('PageRank Vector:')
    print(y_true)
    print()
    y_true = y_true.tolist()
    runs = []
    print('Monte Carlo Simulations:')
    for i in [5, 10, 50, 100]:
        runs.append(monte_carlo(i, i, adj, size))
        print(np.matrix(monte_carlo(i, i, adj, size)).T)
    print()
    measure = []
    for i in range(4):
        total = 0
        for j in range(size):
            x = abs(y_true[j][0] - runs[i][j])
            total += x
        measure.append(total)
    print('L1 errors')
    for i in range(len(measure)):
        print(f'run {i+1} L1 error: {measure[i]}')


def page_rank_vector(matrix, adj, size):
    matrix.pop(0)
    adj.pop(0)
    for i in range(size):
        b = 1 / sum(matrix[i])
        for j in range(size):
            if matrix[i][j] > 0:
                matrix[i][j] = b
    matrix = np.matrix(matrix).T
    p0 = np.matrix([1 / size] * size).T
    p = matrix @ p0
    for i in range(29):
        p = matrix @ p
    return p


def random_index(adj, i, size):
    while True:
        ind = random.randint(0, size - 1)
        if adj[i][ind] == 1:
            return ind


def monte_carlo(m, n, adj, size):
    visits = [0 for i in range(size)]
    for i in range(size):
        for j in range(m):
            k = n - 1
            index = i
            visits[i] += 1
            while k > 0:
                index = random_index(adj, index, size)
                visits[index] += 1
                k -= 1
    PRj = []
    total_visits = size * m * n
    for i in range(size):
        PRj.append(visits[i] / total_visits)
    return PRj


main()
