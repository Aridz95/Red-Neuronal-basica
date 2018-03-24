from net import N
import random as r

                            #   0         1         2         3         4         5         6         7         8         9
training_data = {'inputs': [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1]], 
                            #A B C D E F G   A B C D E F G   A B C D E F G   A B C D E F G   A B C D E F G   A B C D E F G 
                 'target': [[1,1,1,1,1,1,0],[0,1,1,0,0,0,0],[1,1,0,1,1,0,1],[1,1,1,1,0,0,1],[0,1,1,0,0,1,1],[1,0,1,1,0,1,1],
                            #A B C D E F G   A B C D E F G   A B C D E F G   A B C D E F G 
                            [1,0,1,1,1,1,1],[1,1,1,0,0,0,1],[1,1,1,1,1,1,1],[1,1,1,1,1,0,1]]}

nn = N(4,10,7)

print("Salidas sin entrenamiento")
print(nn.predict([0,0,0,0]))
print(nn.predict([0,0,0,1]))
print(nn.predict([0,0,1,0]))
print(nn.predict([0,0,1,1]))
print(nn.predict([0,1,0,0]))
print(nn.predict([0,1,0,1]))
print(nn.predict([0,1,1,0]))
print(nn.predict([0,1,1,1]))
print(nn.predict([1,0,0,0]))
print(nn.predict([1,0,0,1]))

for _ in range(100000):
    index = r.randint(0,9)
    nn.train(training_data['inputs'][index], training_data['target'][index])

print("Salidas con entrenamiento")
print(nn.predict([0,0,0,0]))
print(nn.predict([0,0,0,1]))
print(nn.predict([0,0,1,0]))
print(nn.predict([0,0,1,1]))
print(nn.predict([0,1,0,0]))
print(nn.predict([0,1,0,1]))
print(nn.predict([0,1,1,0]))
print(nn.predict([0,1,1,1]))
print(nn.predict([1,0,0,0]))
print(nn.predict([1,0,0,1]))