import numpy as np

arr = [
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]],
        [[8, 9], [10, 11]]
      ]

tup = tuple(arr)
arr = np.array(arr)

x = arr[:, 0]
y = arr[:][0]

print(x)
print("\n")
print(y)
