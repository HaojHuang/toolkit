import open3d as o3d
from pathlib import Path
import os
import pyrender
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data,Batch,DataLoader
import torch
from typing import Optional, Callable, List

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
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    ax.set_zlim(-0.15, 0.15)

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
    gt, face_index = trimesh.sample.sample_surface(tri_mesh, count=1024*9, face_weight=None)
    return gt

def render_partial(tri_mesh):
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    width = 320
    height = 240
    focal = 120
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


class YCB_Dataset(InMemoryDataset):
    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.root_path = Path(root)
        super(YCB_Dataset,self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self)->str:
        return 'yy.csv'

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt']

    def download(self):
        # Download to `self.raw_dir`.
        print(self.raw_file_names)

    def process(self):
        data_list = []
        for f in (self.root_path / "ycb").iterdir():
            mesh_path = f / 'poisson' / 'textured.obj'
            print(mesh_path)
            if os.path.exists(str(mesh_path)):
                #print(mesh_path)
                tri_mesh = trimesh.load(str(mesh_path))
                # print(tri_mesh)
                tri_mesh = rotate_trimesh(tri_mesh)
                gt = sample_gt(tri_mesh)
                gt = torch.from_numpy(gt).to(torch.float)
                for i in range(10):
                    partial = render_partial(tri_mesh)
                    #print(len(partial))
                    #plot_PCG(partial, gt)
                    #plt.show()
                    partial = torch.from_numpy(partial).to(torch.float)
                    npts = torch.as_tensor(len(partial)).to(torch.long)
                    data = Data(pos=partial,gt=gt,npts=npts)
                    data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        #new_data = Batch.from_data_list(data_list)
        #print(new_data)
        torch.save(self.collate(data_list),self.processed_paths[0])



class YCBNormalize:
    def __init__(self):
        self.center = True
    def __call__(self,data:Data):
        gt = data.gt
        center = torch.mean(gt,dim=0,keepdim=True)
        gt = gt - center
        data.gt = gt
        data.pos = data.pos - center
        return data


# dataset_ycb = YCB_Dataset(root='models')
# loader = DataLoader(dataset_ycb,batch_size=2,shuffle=False)
# print(len(dataset_ycb))
# for batch in loader:
#     print(batch)
#     break
