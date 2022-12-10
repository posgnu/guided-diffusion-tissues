import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 256)
y = np.arange(0, 256)
arr = np.zeros((y.size, x.size))

cx = 128.
cy = 128.
r = 128.

# The two lines below could be merged, but I stored the mask
# for code clarity.
mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 <= (r**2 + 100)
arr[mask] = 123.

# This plot shows that only within the circle the value is set to 123.
plt.figure(figsize=(6, 6))
plt.pcolormesh(x, y, arr)
plt.colorbar()
# import code
# code.interact(local=locals())
plt.show()