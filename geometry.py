import pandas as pd
import numpy as np
import pysph.tools.geometry as geom
import matplotlib.pyplot as plt 

## TEST CASE 1: no slip
# x, y = geom.get_2d_wall(dx = 0.01,center=[0., -0.01], length=1.2)

#TEST CASE 2: No penetration
# y, x = geom.get_2d_wall(dx = 0.01,center=[0.5, 0.0], length=1.2)

#TEST CASE 3: Airfoil with walls
x1, y1 = geom.get_4digit_naca_airfoil(dx=0.01)
x2, y2 = geom.get_2d_wall(dx = 0.01,center=[0.5, -0.01], length=3)
x3, y3 = geom.get_2d_wall(dx = 0.01,center=[0.5, 1.01], length=3)
x = np.concatenate((x3, x2, 0.5 + 0.5 * x1))
y = np.concatenate((y3, y2, 0.5 + 0.5 * y1))
plt.scatter(x, y)
plt.axis('equal')
plt.show()


with open('airfoil_wall.csv', 'w') as f:
    for i in range(len(x)):
        f.write(str(x[i]) + "," + str(y[i]) + "\n")

