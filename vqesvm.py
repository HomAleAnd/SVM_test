#https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02a_training_a_quantum_model_on_a_real_dataset.html
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.utils import algorithm_globals

import numpy as np
# import neal
import dimod
from itertools import product

#Загрузка данных
iris_data = load_iris()
features = MinMaxScaler().fit_transform(iris_data.data) # Нормализация признаков на [0,1]
labels = iris_data.target

algorithm_globals.random_seed = 123 # Seed

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

# QSVC METHOD FROM QISKIT
qsvc = QSVC() # Использование стандартного ядра
qsvc.fit(train_features,train_labels)
print("QSVC Score: ",qsvc.score(test_features,test_labels))

# VQC
from qiskit_machine_learning.algorithms.classifiers import VQC

vqc = VQC(num_qubits=4)

vqc.fit(train_features, train_labels)
train_score_q4 = vqc.score(train_features, train_labels)
test_score_q4 = vqc.score(test_features, test_labels)
print("VQC Score: ", test_score_q4)

# Бинарная классификация
# Используем только два класса (0 и 1), преобразуем метки в -1 и 1

#features=features[labels !=2 ]
#labels=labels[labels !=2 ]
#labels[labels==0]=-1
train_features2, test_features2, train_labels2, test_labels2 = train_test_split(
    features, labels, train_size=0.09, random_state=algorithm_globals.random_seed
)

data = train_features2
t = train_labels2

gamma = 3
N = len(train_features2) # number of training points
K = 2 # number of variables for encoding
C = 3.0 # regularization
B = 2 # base for variable encoding
xi = 0.001 # QUBO penalty
sampler = dimod.SimulatedAnnealingSampler()

def kernel(x, y):
	if gamma == -1:
		k = np.dot(x, y) # scalar product
	elif gamma >= 0:
		k = np.exp(-gamma*(np.linalg.norm(x-y, ord=2))) # Gaussian kernel
	return k

def delta(i, j):
	if i == j:
		return 1
	else:
		return 0
Q_tilde = np.zeros((K*N, K*N))
for n in range(N):
	for m in range(N):
		for k in range(K):
			for j in range(K):
				Q_tilde[(K*n+k, K*m+j)] = 0.5*(B**(k+j))*t[n]*t[m] * \
                            (kernel(data[n], data[m])+xi) - \
                            (delta(n, m)*delta(k, j)*(B**k))

Q = np.zeros((K*N, K*N))
for j in range(K*N):
	Q[(j, j)] = Q_tilde[(j, j)]
for i in range(K*N):
	if i < j:
		Q[(i, j)] = Q_tilde[(i, j)] + Q_tilde[(j, i)]

size_of_q = Q.shape[0]
qubo = {(i, j): Q[i, j] for i, j in product(range(size_of_q), range(size_of_q))} # as a dictionary

response = sampler.sample_qubo(qubo, num_reads=100)
a = response.first.sample
alpha = {}
for n in range(N):
	alpha[n] = sum([(B**k)*a[K*n+k] for k in range(K)])

b = sum([alpha[n]*(C-alpha[n])*(t[n]-(sum([alpha[m]*t[m]*kernel(data[m], data[n]) for m in range(N)]))) for n in range(N)])/sum([alpha[n]*(C-alpha[n]) for n in range(N)])

data2 = test_features2
t2 = test_labels2
#возможно хочет сохранить k^n всех видов
tp,fp,tn,fn=0,0,0,0

for i in range(len(t2)):
	predicted_cls =  sum([alpha[n]*t[n]*kernel(data[n], data2[i]) for n in range(N)]) + b
	y_i = t2[i]
	if(y_i == 1):
		if(predicted_cls > 0):
			tp += 1
		else:
			fp += 1
	else:
		if(predicted_cls < 0):
			tn += 1
		else:
			fn += 1
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f_score = tp/(tp + 1/2*(fp+fn))
accuracy = (tp + tn)/(tp+tn+fp+fn)

print("Binary QUBO Precision, Recall, F Score, Accuracy: ", precision, recall, f_score, accuracy)

from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram

qp = QuadraticProgram()
qp.binary_var_list(K*N)
quadratic = np.zeros((K*N, K*N))
for i in range(K*N):
	for j in range(K*N):
		if i!=j:
			quadratic[(i,j)]=Q_tilde[i,j]
qp.minimize(linear = [Q[i,i] for i in range(K*N)], quadratic = quadratic)
#qubo = QuadraticProgramToQubo().convert(task) #convert to QUBO

n_q = K*N
ry = TwoLocal(n_q, "ry", "cz", reps=5, entanglement="linear")
opt = SPSA(maxiter=10)
sampler=Sampler()
mes = SamplingVQE(sampler=sampler, ansatz=ry, optimizer=opt)
meo = MinimumEigenOptimizer(min_eigen_solver=mes)
result = meo.solve(qp)
a=result.x
alpha2 = {}
for n in range(N):
	alpha2[n] = sum([(B**k)*a[K*n+k] for k in range(K)])

b2 = sum([alpha2[n]*(C-alpha2[n])*(t[n]-(sum([alpha2[m]*t[m]*kernel(data[m], data[n]) for m in range(N)]))) for n in range(N)])/sum([alpha2[n]*(C-alpha2[n]) for n in range(N)])

test_features2=test_features[test_labels !=2 ]
test_labels2=test_labels[test_labels !=2 ]
test_labels2[test_labels2 ==0 ] = -1
data2 = test_features2
t2 = test_labels2

tp,fp,tn,fn=0,0,0,0

for i in range(len(t2)):
	predicted_cls =  sum([alpha2[n]*t[n]*kernel(data[n], data2[i]) for n in range(N)]) + b2
	y_i = t2[i]
	if(y_i == 1):
		if(predicted_cls > 0):
			tp += 1
		else:
			fp += 1
	else:
		if(predicted_cls < 0):
			tn += 1
		else:
			fn += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f_score = tp/(tp + 1/2*(fp+fn))
accuracy = (tp + tn)/(tp+tn+fp+fn)
print("VQE Implementation Metrics: ", precision, recall, f_score, accuracy)




