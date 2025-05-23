# Импорт необходимых библиотек
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.algorithms import QSVC

from qiskit.circuit.library import ZFeatureMap
import matplotlib.pyplot as plt
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Загрузка данных Iris
iris_data = load_iris()
features = MinMaxScaler().fit_transform(iris_data.data)  # Нормализация признаков
labels = iris_data.target

# Разделение данных на обучающую и тестовую выборки
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=42
)

# Создание квантового feature map
feature_map = ZFeatureMap(feature_dimension=4, reps=2)

# Значения gamma для исследования
gamma_values = [0.1, 1, 10, 100]
accuracies = []

# Исследование зависимости точности от gamma
for gamma in gamma_values:
    print(f"Gamma: {gamma}")

    # Создание квантового ядра с текущим gamma
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    # Обучение QSVC с текущим gamma
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(train_features, train_labels)

    # Оценка точности на тестовых данных
    accuracy = qsvc.score(test_features, test_labels)
    accuracies.append(accuracy)
    print(f"Accuracy: {accuracy}")

# Построение графика
plt.plot(gamma_values, accuracies, marker='o')
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Dependence of Accuracy on Gamma')
plt.show()