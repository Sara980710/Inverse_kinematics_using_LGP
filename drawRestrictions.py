import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

L = np.array(
    [0, 0.055, 0.315, 0.045, 0.108, 0.005, 0.034, 0.015, 0.088, 0.204])

# Sphere outer
sphere_radius_outer = np.sqrt(
    (L[6] + L[8] + L[9]) ** 2 + L[4]**2)

max_angle = np.array([np.pi, np.pi, np.pi/2])
min_angle = np.array([0, 0, 0])

# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = sphere_radius_outer*np.cos(u)*np.sin(v)
y = sphere_radius_outer*np.sin(u)*np.sin(v)
z = (L[2] + L[3]) + sphere_radius_outer*np.cos(v)
ax.plot_wireframe(x, y, z, color="r")

# Sphere inner upper
sphere_radius = np.sqrt(np.sqrt(L[8] ** 2 + (L[7] + L[9])**2) ** 2 + (L[4]-L[5])**2)

max_angle = np.array([np.pi, np.pi, np.pi/2])
min_angle = np.array([0, 0, 0])

# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = sphere_radius*np.cos(u)*np.sin(v)
y = sphere_radius*np.sin(u)*np.sin(v)
z = (L[2] + L[3]) + sphere_radius*np.cos(v)
ax.plot_wireframe(x, y, z, color="b")

# Sphere inner lower
sphere_radius = np.sqrt(L[9] ** 2 + (L[4] - L[5]) ** 2)

max_angle = np.array([np.pi, np.pi, np.pi/2])
min_angle = np.array([0, 0, 0])

# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = sphere_radius*np.cos(u)*np.sin(v)
y = sphere_radius*np.sin(u)*np.sin(v)
z = (L[2] + L[3] - L[8]) + sphere_radius*np.cos(v)
ax.plot_wireframe(x, y, z, color="g")

max_pos = np.array([
    sphere_radius_outer,  # x
    sphere_radius_outer,  # y
    L[2] + L[3] + L[8] + L[9]  # z
])  # meter

min_pos = np.array([
    -(L[6] + L[8] + L[9]),  # x
    L[6] - L[7] - L[9],  # y
    L[2] + L[3] - L[8] - L[9]  # z
])  # meter

ax.scatter(max_pos[0], max_pos[1], max_pos[2])
ax.scatter(min_pos[0], min_pos[1], min_pos[2])
ax.scatter(max_pos[0], max_pos[1], min_pos[2])
ax.scatter(max_pos[0], min_pos[1], min_pos[2])
ax.scatter(min_pos[0], max_pos[1], min_pos[2])
ax.scatter(min_pos[0], max_pos[1], max_pos[2])
ax.scatter(min_pos[0], min_pos[1], max_pos[2])
ax.scatter(max_pos[0], min_pos[1], max_pos[2])

ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
ax.set_title(f"Restritions")

plt.show()
