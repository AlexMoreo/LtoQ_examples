import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from quapy.functional import uniform_prevalence_sampling

# Vertices of a regular tetrahedron
v1 = np.array([0, 0, 0])
v2 = np.array([1, 0, 0])
v3 = np.array([0.5, np.sqrt(3)/2, 0])
v4 = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/3])

vertices = np.array([v1, v2, v3, v4])

# Function to map (p1,p2,p3,p4) to 3D coordinates
def prob_to_xyz(p):
    return p[0]*v1 + p[1]*v2 + p[2]*v3 + p[3]*v4

# --- Example: 5 random distributions inside the simplex
rand_probs = uniform_prevalence_sampling(n_classes=4, size=3000)
points_xyz = np.array([prob_to_xyz(p) for p in rand_probs])

# --- Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw tetrahedron faces
faces = [
    [v1, v2, v3],
    [v1, v2, v4],
    [v1, v3, v4],
    [v2, v3, v4]
]

poly = Poly3DCollection(faces, alpha=0.15, edgecolor='k', facecolor=None)
# poly = Poly3DCollection(faces, alpha=0.15, edgecolor='k', facecolor=None)
ax.add_collection3d(poly)

edges = [
    [v1, v2],
    [v1, v3],
    [v1, v4],
    [v2, v3],
    [v2, v4],
    [v3, v4]
]

for edge in edges:
    xs, ys, zs = zip(*edge)
    ax.plot(xs, ys, zs, color='black', linewidth=1)

# Draw vertices
# ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], s=60, color='red')

# Labels
offset = 0.08
labels = ["$y_1$", "$y_2$", "$y_3$", "$y_4$"]
for i, v in enumerate(vertices):
    direction = v / np.linalg.norm(v)
    label_pos = v + offset * direction
    ax.text(label_pos[0], label_pos[1], label_pos[2], labels[i], fontsize=14, color='black')

# Plot random points
ax.scatter(points_xyz[:,0], points_xyz[:,1], points_xyz[:,2], s=10, c='blue', alpha=0.2)

# Axes formatting
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("4-Class Probability Simplex (Tetrahedron)")

# ax.view_init(elev=65, azim=20)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.xaxis.set_ticks_position('none')  # evita que dibuje marcas
ax.yaxis.set_ticks_position('none')
ax.zaxis.set_ticks_position('none')

ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)

ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')

ax.grid(False)

plt.tight_layout()
# plt.show()
plt.savefig('plots/tetrahedron.pdf')
