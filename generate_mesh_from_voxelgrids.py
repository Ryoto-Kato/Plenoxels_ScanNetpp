import svox2, mcubes, torch, numpy, argparse
import numpy as np
import os,sys
from tqdm import trange
path_to_VCAI_libs = os.path.join(os.getcwd(), os.pardir, "VCAI-utils", "libs")
sys.path.append(path_to_VCAI_libs)
from dataset_handler import Filehandler

def generate_mesh(resolution):
    """Generate a mesh grid within (-1, 1) range."""
    range_x = np.linspace(-1, 1, resolution)
    range_y = np.linspace(-1, 1, resolution)
    range_z = np.linspace(-1, 1, resolution)

    mesh_x, mesh_y, mesh_z = np.meshgrid(range_x, range_y, range_z)
    
    return np.vstack((mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten())).T.astype(np.float32)


path2data = "/home/kato/Documents/HiWi/Plenoxels_ScanNetpp/opt/ckpt/samples_maxedge1_json40_scenes"
list_scene_names, list_scene_paths = Filehandler.dirwalker_InFolder(path_to_folder=path2data, prefix='')
for scene_name, scene_path in zip(list_scene_names, list_scene_paths):
    ckpt_path = os.path.join(scene_path, "ckpt.npz")
    targetpath = os.path.join(scene_path,"reconst.obj")
    print("result will be saved to:" , targetpath)
    print("loading sparse grid")
    grid = svox2.SparseGrid.load(ckpt_path)

    reso = 256
    grid_coords = generate_mesh(reso)
    print(np.max(grid_coords, axis = 0))
    print(np.min(grid_coords, axis = 0))
    print(np.mean(grid_coords))
    grid_coords = grid_coords.reshape(8, -1, 3) #for save cpu memory
    density_grid = torch.concat([grid.sample(torch.Tensor(grid_coords[idx]),want_colors=False)[0] for idx in trange(grid_coords.__len__())], dim=0)
    density_grid= density_grid.view(reso, reso, reso).detach().numpy()
    print(np.max(density_grid))
    print(density_grid.shape)
    v,t = mcubes.marching_cubes(density_grid,20) #adjust value to your scene. start with 0.
    print(np.max(v, axis=0))
    print(np.mean(v, axis=0))
    print(np.min(v, axis=0))
    # max, min
    max_coords = np.array([reso-1, reso-1, reso-1])
    min_coords = np.array([0, 0, 0])

    # edges_xyz
    radius_of_aabb = (np.array([max_coords[0]-min_coords[0], max_coords[1]-min_coords[1], max_coords[2]-min_coords[2]]).max())
    print(radius_of_aabb)

    # aligned bbox
    scene_scale = 1/radius_of_aabb # -1, 1

    # scaled the v
    scaled_v = v*scene_scale

    # center the v
    centered_v = scaled_v - np.array([0.5, 0.5, 0.5])[None, :]

    print(np.max(centered_v))
    print(np.min(centered_v))
    # filp y
    second_positive_scale = 0.5/(np.max(centered_v, axis = 0).max())
    second_negative_scale = -0.5/(np.min(centered_v, axis = 0).min())
    second_scale = second_positive_scale if second_positive_scale >= second_negative_scale else second_negative_scale
    print(second_scale)
    centered_v*=second_scale
    mcubes.export_obj(centered_v,t,targetpath)