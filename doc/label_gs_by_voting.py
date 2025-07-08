# Author: Han Huijun
# Date: 2025-01-21
# Last Update: 2025-01-21
# Version: 1.0

# this file identifies the gaussian splatting label 
# by finding the nearest gaussians to the pcd points
# which only includes the interested objects


# output:
# 1. label.npy: the label of each gaussian
# 2. labeled_gs.ply: the visualization of the label

# input:    
# 1. pcd folder: the pcd files of the interested objects
# or pcd folder of the wall and ground

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import numpy as np
import os
from final_eval import load_scene_data
from plyfile import PlyData, PlyElement 
from export_ply import save_ply
from sklearn.neighbors import KDTree
import argparse
from importlib.machinery import SourceFileLoader
from tqdm import tqdm
import json

RGB_LIST = [
    [0, 0, 255], # blue 0
    [255, 255, 0], # green 1
    [255, 0, 255], 
   
    [0, 255, 255],  # cyan 3

     [255, 0, 0], # red 2  
    [0, 128, 128],
    [0, 0, 128],   # dark blue 6
    [0, 128, 0],  
    [128, 0, 0],  
    [240, 240, 240], # yellow 5
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


if __name__ == "__main__":

    
    go_to_read_npy = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help="Path to experiment file", default="./configs/iphone/nerfcapture_off.py")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.config), args.config
    ).load_module()

    config = experiment.config
    workdir = config['workdir']

    # input (to be modified)
    obj_type = "furniture"
    pcd_path = os.path.join(workdir, config['run_name'], "eval_train", "pcd", obj_type)



    # input (fixed)
    scene_path = os.path.join(workdir, config['run_name'], "params.npz")

    # to visualize the result 
    # ply_path = os.path.join(workdir, config['run_name'], "eval", "labeled_gs.ply")
    ply_path = os.path.join(workdir, config['run_name'], "eval", "labeled_gs_2.ply")
    
    # output: the label
    label_path = os.path.join(workdir, config['run_name'], "eval", "label.npy")
    floor_path = os.path.join(workdir, config['run_name'], "eval", "floor.npy")
    wall_path = os.path.join(workdir, config['run_name'], "eval", "wall.npy")

    


    pcd_folders = os.listdir(pcd_path)
    max_frame = 0
    frame_id_to_folder = {}

    # load gaussian Splating
    params = load_scene_data(scene_path)
    means = params['means3D'].detach().cpu().numpy()
    scales = params['log_scales'].detach().cpu().numpy()
    rotations = params['unnorm_rotations'].detach().cpu().numpy()
    rgbs = params['rgb_colors'].detach().cpu().numpy()
    opacities = params['logit_opacities'].detach().cpu().numpy()

 


    category = 0
    folder_lut = {}
    folder_lut_reverse = {}
    # record the "frame id" to "label" mapping
    if obj_type == "furniture":
        for folder in pcd_folders:
            if not os.path.isdir(os.path.join(pcd_path, folder)):
                continue
            
            pcd_files = os.listdir(os.path.join(pcd_path, folder))
            for pcd_file in pcd_files:
                frame = int(pcd_file.split(".")[0].split("_")[-1])
                if frame not in frame_id_to_folder:
                    frame_id_to_folder[frame] = [folder]
                else:
                    frame_id_to_folder[frame].append(folder)
                folder_lut[folder] = category
                folder_lut_reverse[category] = folder
            category += 1
    elif obj_type == "wall":
        folder_lut_reverse[category] = "1"
        # all the files has the same category -> wall
        for files in pcd_folders:
            frame = int(files.split(".")[0].split("_")[-1])
            if frame not in frame_id_to_folder:
                frame_id_to_folder[frame] = [files]
            else:
                frame_id_to_folder[frame].append(files)
            folder_lut[files] = category
        category += 1

    folder_lut_reverse[category] = "-1"


    # establish the label count table
    # every frame is a voter, vote for the label of each gaussian
    # in other words, each gaussian row has a vote 
    label_count =np.zeros((means.shape[0], category + 1))


    vote_frame = 0
    # Structure of the algorithm
    # OUTER LOOP: for each frame
    # INNER LOOP: for each folder in the frame
    
    # frame_id_to_folder: {frame_id: [folder1, folder2, folder3]}
    # so it is an orderless dict
    for frame in tqdm(frame_id_to_folder):
        if go_to_read_npy:
            break
        total_xyz = []
        total_label = []
        for folder in frame_id_to_folder[frame]:
            if obj_type == "furniture":
                print(f"Frame {frame} in folder {folder}")
            elif obj_type == "wall":
                print(f"Wall annotation: processing Frame {frame}")
            # load the pcd file
            if obj_type == "furniture":
                pcd_file = os.path.join(pcd_path, folder, f"pcd_{frame}.ply")
            elif obj_type == "wall" or obj_type =="chair":
                pcd_file = os.path.join(pcd_path, f"pcd_{frame}.ply")

            pcd = PlyData.read(pcd_file)
            xyz = np.array([list(x) for x in pcd.elements[0]])
            xyz = xyz[:, :3]
            total_xyz.append(xyz)

            # construct the label 
            # by [folder_name_0] * num_points +  [folder_name_1] * num_points + ...
            label = np.ones(xyz.shape[0]) * folder_lut[folder]

            total_label.append(label)

        total_xyz = np.concatenate(total_xyz, axis=0)
        total_label = np.concatenate(total_label, axis=0)

        tree = KDTree(total_xyz)

        # Each gaussian find 1 nearest point, using its label to label itself
        dist, ind = tree.query(means, k=1)
        mask = np.exp(scales) < dist

        # Assign the label to the gaussians
        gauss_label = total_label[ind].reshape(-1)
        # if the distance is too far, then it is a dustbin
        gauss_label[mask.reshape(-1)] = category

        # from label to a 2D table of label count 
        # label_count: 2D table
        # row: gaussians, column: category + 1 dustbin
        # determine which cell to increment

        table_width = category + 1
        table_height = means.shape[0]

        ind_in_label_count = np.arange(table_height) * table_width +  gauss_label # index of the cell to increment
        ind_in_label_count = ind_in_label_count.astype(int)
        label_count.reshape(-1)[ind_in_label_count] += 1


        vote_frame += 1
    # for each gaussian find its label
    print("label count")
    for i in range(category+1):
        category_name = folder_lut_reverse[i]
        print(f"{category_name}: {label_count[:,i].sum()}")

    # the last column is the dustbin, so we find the max in the rest columns
    label = np.argmax(label_count[:,:-1], axis=1) 

    # if dustbin gets all the votes, then the label is category
    label[label_count[:,-1]==vote_frame] = category

    my_label = np.array(list(folder_lut_reverse.values()))[label]
    
    # if not go_to_read_npy:
    #     np.save(label_path, my_label)
    # else:
    #     my_label = np.load(label_path)
    np.save(label_path, my_label)

    lut = {4:16, 16:16,5:22,11:22,22:22,9:15,15:15,7:14,14:14}



    

        
    my_floor = np.load(floor_path)
    my_wall = np.load(wall_path)

    for i in range(len(my_label)):
        # if my_wall[i] == '1' or my_floor[i] == '1':
        #     # if my_floor[i] == '1':
        #     #     my_label[i] = '1'
        #     # else:
        #     #     my_label[i] = '2'
        #     # continue

        if int(my_label[i]) in lut:
            pass
            my_label[i] = str(lut[int(my_label[i])])
        else:
            my_label[i] = '-1'

    # draw the color on the point cloud
    rgbs = np.array([RGB_LIST[int(i)%RGB_len] for i in my_label])
    save_ply(ply_path, means, scales, rotations, rgbs, opacities)

    # read json
    instance_info = {}
    instance_new_info = {}
    instance_info_path = os.path.join(workdir, "label.json") # workdir is the capture date timestamp
    verbose_info_path = os.path.join(workdir, "verbose_label.json")
    if os.path.exists(instance_info_path):
        with open(instance_info_path, 'r') as f:
            instance_info = json.load(f)
    else:
        print("No instance info json file found")
    for k, v in instance_info.items():
        label_k = folder_lut[k]
        index_label = label == label_k
        num_points = int( np.sum(index_label))
        index_list = np.arange(len(index_label))[index_label]
        index_list = [int(i) for i in index_list]
        instance_new_info[k] =  build_instance_dict(v, index_list, num_points)
    with open(verbose_info_path, 'w') as f:
        json.dump(instance_new_info, f, indent=4)
    
        
    

    









