# draw a sin and save to the current directory

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

print('Test output 1')


plt.plot(x, y)
plt.savefig('sin.png')

# response = CALL_API