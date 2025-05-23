from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.utils import algorithm_globals
import numpy as np
import dimod
from itertools import product
import time
from memory_profiler import memory_usage
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

def main():
    # Загрузка данных
    iris_data = load_iris()
    features = MinMaxScaler().fit_transform(iris_data.data)  # Нормализация признаков на [0,1]
    labels = iris_data.target

    algorithm_globals.random_seed = 123  # Seed

    class1 = 1  # Первый класс
    class2 = 2  # Второй класс

    # Фильтрация данных по выбранным классам
    mask = (labels == class1) | (labels == class2)  # Оставляем только выбранные классы
    features = features[mask]  # Перезаписываем features
    labels = labels[mask]  # Перезаписываем labels

    # Преобразование меток для бинарной классификации
    labels[labels == class1] = -1
    labels[labels == class2] = 1

    # Разделение на обучающих и тестовых данных
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, train_size=0.09, random_state=algorithm_globals.random_seed
    )

    data = train_features
    t = train_labels

    gamma = 3
    N = len(train_features)  # number of training points
    K = 2  # number of variables for encoding
    C = 3.0  # regularization
    B = 2  # base for variable encoding
    xi = 0.001  # QUBO penalty
    sampler = dimod.SimulatedAnnealingSampler()

    def kernel(x, y):
        if gamma == -1:
            k = np.dot(x, y)  # scalar product
        elif gamma >= 0:
            k = np.exp(-gamma * (np.linalg.norm(x - y, ord=2)))  # Gaussian kernel
        return k

    def delta(i, j):
        if i == j:
            return 1
        else:
            return 0

    Q_tilde = np.zeros((K * N, K * N))
    for n in range(N):
        for m in range(N):
            for k in range(K):
                for j in range(K):
                    Q_tilde[(K * n + k, K * m + j)] = 0.5 * (B ** (k + j)) * t[n] * t[m] * \
                                                      (kernel(data[n], data[m]) + xi) - \
                                                      (delta(n, m) * delta(k, j) * (B ** k))

    Q = np.zeros((K * N, K * N))
    for j in range(K * N):
        Q[(j, j)] = Q_tilde[(j, j)]
    for i in range(K * N):
        if i < j:
            Q[(i, j)] = Q_tilde[(i, j)] + Q_tilde[(j, i)]

    size_of_q = Q.shape[0]
    qubo = {(i, j): Q[i, j] for i, j in product(range(size_of_q), range(size_of_q))}  # as a dictionary

    response = sampler.sample_qubo(qubo, num_reads=100)
    a = response.first.sample
    alpha = {}
    for n in range(N):
        alpha[n] = sum([(B ** k) * a[K * n + k] for k in range(K)])

    b = sum([alpha[n] * (C - alpha[n]) * (t[n] - (sum([alpha[m] * t[m] * kernel(data[m], data[n]) for m in range(N)]))) for n in range(N)]) / sum([alpha[n] * (C - alpha[n]) for n in range(N)])

    data2 = test_features
    t2 = test_labels
    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(t2)):
        predicted_cls = sum([alpha[n] * t[n] * kernel(data[n], data2[i]) for n in range(N)]) + b
        y_i = t2[i]
        if y_i == 1:
            if predicted_cls > 0:
                tp += 1
            else:
                fp += 1
        else:
            if predicted_cls < 0:
                tn += 1
            else:
                fn += 1

    # Создание QuadraticProgram
    qp = QuadraticProgram()
    qp.binary_var_list(K * N)
    quadratic = np.zeros((K * N, K * N))
    for i in range(K * N):
        for j in range(K * N):
            if i != j:
                quadratic[(i, j)] = Q_tilde[i, j]
    qp.minimize(linear=[Q[i, i] for i in range(K * N)], quadratic=quadratic)

    # Преобразование QuadraticProgram в QUBO
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)

    # Преобразование QUBO в оператор Изинга
    operator, offset = qubo.to_ising()

    # Настройка VQE
    n_q = K * N
    ry = TwoLocal(n_q, "ry", "cz", reps=3, entanglement="linear")
    opt = SPSA(maxiter=10)
    sampler = Sampler()

    # Измерение времени и памяти для VQE
    start_time = time.time()
    mem_usage = memory_usage((SamplingVQE(sampler=sampler, ansatz=ry, optimizer=opt).compute_minimum_eigenvalue, (operator,)))
    mes = SamplingVQE(sampler=sampler, ansatz=ry, optimizer=opt)
    result = mes.compute_minimum_eigenvalue(operator)
    end_time = time.time()
    elapsed_time = end_time - start_time
    max_mem_usage = max(mem_usage)

    # Получение результата
    a = result.eigenstate
    alpha2 = {}
    for n in range(N):
        alpha2[n] = sum([(B ** k) * a[K * n + k] for k in range(K)])

    b2 = sum([alpha2[n] * (C - alpha2[n]) * (t[n] - (sum([alpha2[m] * t[m] * kernel(data[m], data[n]) for m in range(N)]))) for n in range(N)]) / sum([alpha2[n] * (C - alpha2[n]) for n in range(N)])

    test_features2 = test_features[test_labels != 2]
    test_labels2 = test_labels[test_labels != 2]
    test_labels2[test_labels2 == 0] = -1
    data2 = test_features2
    t2 = test_labels2

    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(t2)):
        predicted_cls = sum([alpha2[n] * t[n] * kernel(data[n], data2[i]) for n in range(N)]) + b2
        y_i = t2[i]
        if y_i == 1:
            if predicted_cls > 0:
                tp += 1
            else:
                fp += 1
        else:
            if predicted_cls < 0:
                tn += 1
            else:
                fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = tp / (tp + 0.5 * (fp + fn))
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("VQE Implementation Metrics: ", precision, recall, f_score, accuracy)
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Maximum memory usage: {max_mem_usage:.2f} MiB")

if __name__ == '__main__':
    main()