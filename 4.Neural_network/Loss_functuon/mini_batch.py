# 머신러닝에서의 batch란? 
# 모델을 학습할 때 1회 반복(이터레이션) 당 사용되는 데이터셋의 모임
# 1회 반복은 학습을 반복하는 횟수를 말한다. 
# 신경망은 훈련데이터를 사용하여 학습하는데 구체적으로는 
# 훈련데이터에 대한 손실함수 값을 구하고 그 값을 최대한 줄여주는 매개변수를 찾아낸다
# 따라서 훈련데이터가 n 개라면 각 훈련 데이터에 대한 손실함수 값의 평균을 구한다. 
# 하지만 이러한 방법은 각각으 데이터에 대해 손실함수를 구해야 하기 때문에 데이터의 양이
# 많아지면 오래걸릴 수 있다.따라서 이러한 경우에는 데이터 일부를 추려서 전체의 근사치로 이용할 수 있는데
# 이 일부를 미니배치라고 한다. 가령 m 개의 데이터가 있다고 하면 그중 n 개를 무작위로 뽑아 훈련하는것을 
# 미니배치 학습이라고 한다. 
#예시 

import sys, os 
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize = True, one_hot_label = True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
# 60000 개 데이터 중 10개를 무작위로 선택
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
