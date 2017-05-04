
import numpy as np
import matplotlib.pyplot as plt
# Compute the x and y coordinates for points on sine and cosine
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2, 1, 1)
# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')
# Set the second subplot as active, and make the second plot.
plt.subplot(212)
plt.plot(x, y_cos)
plt.title('Cosine')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['practical'])
# Show the figure.
plt.grid(True)
plt.show()
