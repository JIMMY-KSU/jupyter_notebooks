
# coding: utf-8

# In[1]:

get_ipython().magic('pylab')
import yt
from mpl_toolkits.mplot3d import Axes3D


# In[3]:

ds = yt.load('MOOSE_sample_data/mps_out.e', step=-1)


# In[4]:

index = 6899  # selects an element
m = ds.index.meshes[1]
coords = m.connectivity_coords
indices = m.connectivity_indices - 1
vertices = coords[indices[index]]


# In[5]:

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

axis = [0, 1, 1]
theta = 1.2

vertices -= vertices.mean(axis=0)
vertices *= 5e3

for i in range(20):
    vertices[i] = np.dot(rotation_matrix(axis,theta), vertices[i]) 


# In[6]:

def patchSurfaceFunc(u, v, verts):
    return 0.25*(1.0 - u)*(1.0 - v)*(-u - v - 1)*verts[0] +            0.25*(1.0 + u)*(1.0 - v)*( u - v - 1)*verts[1] +            0.25*(1.0 + u)*(1.0 + v)*( u + v - 1)*verts[2] +            0.25*(1.0 - u)*(1.0 + v)*(-u + v - 1)*verts[3] +            0.5*(1 - u)*(1 - v*v)*verts[4] +            0.5*(1 - u*u)*(1 - v)*verts[5] +            0.5*(1 + u)*(1 - v*v)*verts[6] +            0.5*(1 - u*u)*(1 + v)*verts[7]


# In[7]:

faces = [[0, 1, 5, 4, 12, 8, 13, 16],
        [1, 2, 6, 5, 13, 9, 14, 17],
        [3, 2, 6, 7, 15, 10, 14, 18],
        [0, 3, 7, 4, 12, 11, 15, 19],
        [4, 5, 6, 7, 19, 16, 17, 18],
        [0, 1, 2, 3, 11, 8, 9, 10]]


# In[8]:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for face in faces:
    N = 16
    x = np.empty(N**2)
    y = np.empty(N**2)
    z = np.empty(N**2)
    for i, u in enumerate(np.linspace(-1.0, 1.0, N)):
        for j, v in enumerate(np.linspace(-1.0, 1.0, N)):
            index = i*N+j
            x[index], y[index], z[index] = patchSurfaceFunc(u, v, vertices[face])
    ax.plot_trisurf(x, y, z)


# In[ ]:



