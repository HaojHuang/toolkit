import open3d as o3d
import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def random_pose():
    angle_x = np.random.uniform() * 2 * np.pi
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(0.4*R[:, 2], 1)
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    return pose

def depth2pcd(depth, intrinsics, pose):
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)
    y, x = np.where(depth > 0)
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0))
    # camera coordinates -> world coordinates
    points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0)).T[:, :3]
    return points

def plot_pcd(ax, pcd):
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=pcd[:, 0], s=0.5, cmap='Reds', vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-0.1, 0.1)

def plot_PCG(partial, gt):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')
    plot_pcd(ax, partial)
    ax.set_title('partial input',fontsize=18)
    ax = fig.add_subplot(122, projection='3d')
    plot_pcd(ax, gt)
    ax.set_title('gt',fontsize=18)
    plt.tight_layout()

def rotate_trimesh(tri_mesh):
    angle_x = np.pi/2
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    t = np.zeros((3, 1))
    r_pose = np.concatenate([np.concatenate([Rx, t], 1), [[0, 0, 0, 1]]], 0)
    return tri_mesh.apply_transform(r_pose)

def sample_gt(tri_mesh):
    gt, face_index = trimesh.sample.sample_surface(tri_mesh, count=8192, face_weight=None)
    return gt

def render_partial(tri_mesh):
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    width = 160
    height = 120
    focal = 100
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=width / 2, cy=height / 2)
    camera_pose = random_pose()
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(width, height)
    color, depth = r.render(scene)
    points = depth2pcd(depth, intrinsics, camera_pose)
    return points

file = './textured.obj'
mesh = o3d.io.read_triangle_mesh(file)
# visualize the mesh from open3d
#o3d.visualization.draw_geometries([mesh],)

# load mesh with trimesh
can_mesh = trimesh.load(file)
#can_mesh.show()

trans,pro = can_mesh.compute_stable_poses()
initial_pose = np.eye(4)
angle_x = -np.pi/2
Rx = np.array([[1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)]])

t = np.zeros((3,1))
r_pose = np.concatenate([np.concatenate([Rx, t], 1), [[0, 0, 0, 1]]], 0)
#print(r_pose)
can_mesh = can_mesh.apply_transform(r_pose)
# uniformly sample 8192 points from the mesh
gt,face_index = trimesh.sample.sample_surface(can_mesh, count=8192, face_weight=None)
print('===============',gt.shape)

#load the mesh to pyrender
can_mesh = pyrender.Mesh.from_trimesh(can_mesh)
scene = pyrender.Scene()
scene.add(can_mesh)
width = 640
height = 480
focal = 150
intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
#camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
camera = pyrender.IntrinsicsCamera(fx=focal,fy=focal,cx=width/2,cy=height/2)
#s = np.sqrt(2)/2
# camera_pose = np.array([
#     [0.0, -s,   s,   0.3],
#     [1.0,  0.0, 0.0, 0.0],
#     [0.0,  s,   s,   0.35],
#     [0.0,  0.0, 0.0, 1.0],])

camera_pose = random_pose()
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,innerConeAngle=np.pi/16.0,outerConeAngle=np.pi/6.0)
scene.add(light,pose=camera_pose)
r= pyrender.OffscreenRenderer(width,height)
color,depth = r.render(scene)

print(depth,depth.shape)

points = depth2pcd(depth,intrinsics,camera_pose)
partial = o3d.geometry.PointCloud()
partial.points = o3d.utility.Vector3dVector(points)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.00, 0.00, 0.00])
compelte = o3d.geometry.PointCloud()
compelte.points = o3d.utility.Vector3dVector(gt)
o3d.visualization.draw_geometries([partial,mesh_frame],)
plot_PCG(points,gt)
plt.show()
