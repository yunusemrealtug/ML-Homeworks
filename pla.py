import numpy as np

data = np.load('data_small.npy')
label = np.load('label_small.npy')



t = 0
wt = np.zeros(3)
w =  wt.reshape((3, 1))

while True:
    broken=False
    for i in range(len(data)):
        if label[i] * np.dot(w.T, data[i]) <= 0:
            w += label[i] * data[i].reshape((3, 1))
            broken = True
            break
    if not broken:
        break
result=w.reshape(3)
print(np.dot(data, result))
print(data)
print(label)




    