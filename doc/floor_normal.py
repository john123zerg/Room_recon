# Author: Han Huijun
# Date: 2025-01-21
# Last Update: 2025-01-29
# Version: 1.0

# this file put label on the Gaussian points,
# by finding the nearest gaussians to the pcd points, that only includes the interested objects

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import numpy as np
from pathlib import Path
import os
from final_eval import load_scene_data
from plyfile import PlyData, PlyElement 
from export_ply import save_ply
from sklearn.neighbors import KDTree
import argparse
from importlib.machinery import SourceFileLoader
from tqdm import tqdm
import json
import open3d as o3d
from utils import object_fitting

RGB_LIST = [
    [0, 0, 255], # blue 0
    [0, 255, 0], # green 1
    [255, 0, 0], # red 2  
    [0, 255, 255],  # cyan 3
    [255, 0, 255], 
    [255, 255, 0], # yellow 5
    [0, 0, 128],   # dark blue 6
    [0, 128, 0],  
    [128, 0, 0],  
    [0, 128, 128],
]
RGB_LIST = np.array(RGB_LIST) / 255.0
RGB_len = len(RGB_LIST)




def build_instance_dict( catogory_name ,  ply_index, num_points,instance_name = "",):
    instance_dict = {
        "category": catogory_name,
        "instance_name": instance_name,
        "ply_index": ply_index,
        "num_points": num_points,
    }
    return instance_dict



# So, I need to clarity: 
# The data structure is that each parent folder is a category
# this is to open  items of this category all at once
# but this cause the coding to be a little bit complex


def fit_floor_and_save_ply(floor_pts, all_pts, result_dir):
    plane = object_fitting.fit_plane(floor_pts, all_pts, all_pts,result_dir=result_dir, visualize=True)
    plane_serializable = {
        'normal': plane["normal"].tolist(),
        'centroid': plane["centroid"].tolist(),
        'boundary_points': plane["boundary_points"].tolist(),
        'v1': plane["v1"].tolist(),
        'v2': plane["v2"].tolist(),
    }
    json.dump(plane_serializable, open(result_dir / "plane.json", "w"))
    return plane

def fit_wall_and_save_ply(plane,wall_pts, result_dir):
    wall = object_fitting.fit_wall_ortho(plane, wall_pts, result_dir = result_dir, visualize=True)
    wall_serializable = {
        "offsetX": wall.offsetX,
        "offsetY": wall.offsetY,
        'factorX': wall.factorX,
        'factorY': wall.factorY,
        'imgpath': str(wall.imgpath),
        'wall_height': wall.wall_height,
    }
    json.dump(wall_serializable, open(result_dir / "wall.json", "w"))
    return wall

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path to experiment file", default="./configs/iphone/nerfcapture_off.py")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.config), args.config
    ).load_module()

    config = experiment.config
    workdir = config['workdir']
    scene_path = os.path.join(workdir, config['run_name'], "params.npz")


    # input: floor, chair, wall
    floor_flag_path = os.path.join(workdir, config['run_name'], "eval", "floor.npy")
    floor_ply_path = os.path.join(workdir, config['run_name'], "eval", "floor.ply")

    chair_flag_path = os.path.join(workdir, config['run_name'], "eval", "chair.npy")
    chair_ply_folder = os.path.join(workdir, config['run_name'], "eval", "chair")
    
    os.makedirs(chair_ply_folder, exist_ok=True)

    wall_flag_path = os.path.join(workdir, config['run_name'], "eval", "wall.npy")

    principal_dir = os.path.join(
        workdir, experiment.config["run_name"], "eval", "wall","wall_principal_dir.json"
    )
    correct_principal = False
    if os.path.exists(principal_dir):
        print("Second run, with correct principal direction")
        correct_principal = True
        wall_principal = json.load(open(principal_dir, "r"))


    

    floor_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"], "eval", "floor"
    )
    os.makedirs(floor_dir, exist_ok=True)
    floor_dir = Path(floor_dir)

    wall_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"], "eval", "wall"
    )
    os.makedirs(wall_dir, exist_ok=True)
    wall_dir = Path(wall_dir)


    # load gaussian Splating
    params = load_scene_data(scene_path)
    all_pts = params['means3D'].detach().cpu().numpy()



    floor_flag = np.load(floor_flag_path)
    flag = floor_flag == '1'
    print("# floor pts:", sum(flag))
    floor_pts = all_pts[flag]

    wall_flag = np.load(wall_flag_path)
    flag = wall_flag == '1'
    print("# wall pts:", sum(flag))
    wall_pts = all_pts[flag]

    # chair_list = ['4','5', '7', '9','11','14','15','16','22']
    chair_list = ['6'] # for testing
    chair_pts_list = []
    for chair_id in chair_list:
        chair_flag = np.load(chair_flag_path)
        chair_flag = chair_flag == chair_id
        selected_means = all_pts[chair_flag]
        o3d.io.write_point_cloud(chair_ply_folder + f'/{chair_id}.ply', o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected_means)))
        chair_pts_list.append(selected_means)




    plane = fit_floor_and_save_ply(floor_pts, all_pts, result_dir = floor_dir)
    wall = fit_wall_and_save_ply (plane, wall_pts, result_dir = wall_dir)

    grid_length = 0.2
    grid_num = 20
    

    if correct_principal:
        occupid_grid_list = []
        object_fitting.draw_grid(wall_principal["principal_direction"],wall_principal["origin"],grid_num,grid_length,result_dir= floor_dir) 
        for chair_pts, id in zip(chair_pts_list,chair_list):
            occupid_grid = object_fitting.mark_occupied_grid(chair_pts, id,wall_principal["principal_direction"],wall_principal["origin"],grid_num,grid_length,result_dir= floor_dir)
            occupid_grid_list.append(occupid_grid)
        # if has same true occupied grid, then cluster them together
        group = {}
        count = 0
        for i in range(len(occupid_grid_list)):
            for j in range(i+1,len(occupid_grid_list)):
                both_occupied = occupid_grid_list[i] & occupid_grid_list[j]
                if both_occupied.any():
                    occupid_grid_list[i] = occupid_grid_list[i] | occupid_grid_list[j]
                    occupid_grid_list[j] = occupid_grid_list[i]
                    # find i's count and j's if no, add to group
                    id_i = chair_list[i]
                    id_j = chair_list[j]
                    if id_i in group:
                        group[id_j] = group[id_i]
                    elif id_j in group:
                        group[id_i] = group[id_j]
                    else:
                        group[id_i] = count
                        group[id_j] = count
                        count += 1
        # write to json
        json.dump(group, open(floor_dir / "group.json", "w"))

    
