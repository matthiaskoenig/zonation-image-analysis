from zia.console import console

from matplotlib import pyplot as plt
from nanomesh import Image
import numpy as np

dab_path = 'dab_test_L7.npy'
dab_data = np.load(dab_path)

dab_data2d = np.zeros(shape=(dab_data.shape[0], dab_data.shape[1], 2))
dab_data2d[:, :, 1] = dab_data[:, :, 0]

plane = Image(dab_data)

console.print(plane)

plane.show()
plt.show()

# Generate the mesh
console.rule(title="mesh generation", style="white")
# In this example, we use q30 to generate a quality mesh
# with minimum angles of 30°, and a50 to limit the triangle size to 50 pixels.
mesh = plane.generate_mesh(opts='q30a10')
console.print(mesh)
