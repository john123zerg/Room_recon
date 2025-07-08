import json
import argparse
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial import ConvexHull,cKDTree
# from vis_utils import build_cylinders, estimate_iou_3d
import open3d as o3d
import open3d.core as o3c
from random import choice
from tqdm import trange
import igl
import time
import pickle
from plyfile import PlyData, PlyElement
from trimesh.transformations import quaternion_from_matrix  # 추가: 쿼터니언 변환 함수 import

    

def prepare_models(taxonomy_path: str, target_categories: dict):
    
    with open(taxonomy_path, 'r') as f:
        taxonomy = json.load(f)

    # print(taxonomy)

    for category in taxonomy:
        print(category['metadata']['label'], category['metadata']['name'])

        for target in target_categories.keys():
            
            if target in category['metadata']['label']:
                target_categories[target]['shapenet_name'] = category['metadata']['name']
    
    target_categories['desk']['shapenet_name'] = target_categories['table']['shapenet_name']


    return target_categories


def load_data(scene_dir: Path, target_label_ids: list, result_dir: Path, conf_threshold: float=0.04):

    scene_pcd = trimesh.load(scene_dir / 'input.ply', maintain_order=True, process=False)

    with open(scene_dir / 'pred_instance' / 'all_instances.txt', 'r') as f:
        all_instances = f.readlines()
    
    target_instances = []
    for obj_instance in all_instances:
        obj_instance = obj_instance.strip().split(' ')
        mask_path = obj_instance[0]
        label_id = int(obj_instance[1])
        conf = float(obj_instance[2])

        if label_id in target_label_ids and conf >= conf_threshold:
            print('Label ID:', label_id, 'Conf:', conf)
            with open(scene_dir / 'pred_instance' / mask_path, 'r') as f:
                mask = f.readlines()
            mask = np.array([int(m.strip()) for m in mask]).astype(bool)
            pts = scene_pcd[mask]
            target_instances.append({
                'pts': pts,
                'label_id': label_id,
            })

            print(mask_path, label_id)

    
    result_dir.mkdir(exist_ok=True)
    target_union = []
    for idx, target_instance in enumerate(target_instances):
        pts = target_instance['pts']
        label_id = target_instance['label_id']
        pcd = trimesh.PointCloud(pts, colors=np.random.rand(3))
        pcd.export(result_dir / f'{label_id}_{idx}_original.ply')
        target_union.append(pcd)

    # target_union = trimesh.util.concatenate(target_union)
    # target_union.export(result_dir / 'target_original_union.ply')

    return target_instances


def replace_by_models(floor_plane, cad_mesh, target_pts, result_dir: Path, iou_weight: float=1e-1):

    # o3d remove pointcloud outlier
    # Vector3dVector
    target_pts = o3d.utility.Vector3dVector(target_pts)
    pcd = o3d.geometry.PointCloud(target_pts)
    # pcd.points = o3d.core.Tensor(, dtype=o3c.float32)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.1)
    pcd = pcd.select_by_index(ind)
    # save pcd to ply
    # o3d.io.write_point_cloud(result_dir / 'target.ply', pcd)
    target_pts = np.asarray(pcd.points)
    # trimesh.PointCloud(target_pts, colors=[0, 255, 0]).export(result_dir / 'target.ply')

    # estimate 3D bounding box

    # target_info_list = []
    # bbox_union = []
    floor_plane['normal'] = np.array(floor_plane['normal'])
    # project to normal
    pts = target_pts
    # centroid = np.mean(pts, axis=0)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    centroid = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
    pts_ = pts - centroid
    pts_height = np.dot(pts_, floor_plane['normal'])
    h_max = np.percentile(pts_height, 98)
    h_min = -np.dot(centroid - floor_plane['centroid'], floor_plane['normal'])

    # project object to floor
    pts_ = pts_ - np.dot(pts_, floor_plane['normal'])[:, None] * floor_plane['normal'][None, :]

    # main axis
    cov = np.cov(pts_, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov)
    v1 = eig_vec[:, np.argmax(eig_val)]
    v2 = np.cross(v1, floor_plane['normal'])

    v1_dist = np.dot(pts_, v1)
    v2_dist = np.dot(pts_, v2)
    v1_min, v1_max = np.percentile(v1_dist, 3), np.percentile(v1_dist, 98)
    v2_min, v2_max = np.percentile(v2_dist, 3), np.percentile(v2_dist, 98)
    
    sizes = np.array([
        v1_max - v1_min,
        v2_max - v2_min,
        h_max - h_min,
    ])
    # sizes *= 1.2  # magic number
    bbox = np.array([
        centroid + v1_min * v1 + v2_min * v2 + h_min * floor_plane['normal'],
        centroid + v1_max * v1 + v2_min * v2 + h_min * floor_plane['normal'],
        centroid + v1_max * v1 + v2_max * v2 + h_min * floor_plane['normal'],
        centroid + v1_min * v1 + v2_max * v2 + h_min * floor_plane['normal'],
        centroid + v1_min * v1 + v2_min * v2 + h_max * floor_plane['normal'],
        centroid + v1_max * v1 + v2_min * v2 + h_max * floor_plane['normal'],
        centroid + v1_max * v1 + v2_max * v2 + h_max * floor_plane['normal'],
        centroid + v1_min * v1 + v2_max * v2 + h_max * floor_plane['normal']
    ])
    
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7]
    ])

    # bbox_pcd = trimesh.PointCloud(bbox, colors=[255, 0, 0])
    # bbox_pcd.export(result_dir / f'{label_id}_{target_idx}_bbox.ply')

    # sticks = []
    # for edge in edges:
    #     stick = build_cylinders(bbox[edge[0]], bbox[edge[1]], 0.01, (0, 255, 0))
    #     sticks.append(stick)
    # sticks = trimesh.util.concatenate(sticks)
    # bbox_union.append(sticks)
    # sticks.export(result_dir / f'{label_id}_{target_idx}_bbox.ply')

    target_info = {
        'centroid': centroid,
        'pts': pts,
        'bbox': bbox,
        'sizes': sizes,
        'v1': v1,
        'v2': v2,
        'h_min': h_min,
    }

    # move to origin
    v = np.array(cad_mesh.vertices)
    x_min, x_max = v[:, 0].min(), v[:, 0].max()
    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    v = v - np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
    # v = v - np.mean(v, axis=0)

    # rotate from y-up to plane normal
    y = np.array([0, 1, 0])
    axis = np.cross(y, floor_plane['normal'])
    angle = np.arccos(np.dot(y, floor_plane['normal']))
    rot_mat = trimesh.transformations.rotation_matrix(angle, axis)
    v = np.dot(v, rot_mat[:3, :3].T)

    # scale to object size
    min_angle_dist = None
    best_v = None 
    best_angle = None
    best_rot_mat = None  # 추가: best rotation matrix 저장
    for angle in range(0, 360, 5):
        axis = floor_plane['normal']
        rot_mat = trimesh.transformations.rotation_matrix(np.radians(angle), axis)
        v_rot = np.dot(v, rot_mat[:3, :3].T)

        # project object to floor
        h = np.dot(v_rot, floor_plane['normal'])
        h_min, h_max = h.min(), h.max()
        # h_min = np.percentile(h, 5)
        # h_max = np.percentile(h, 95)
        v_rot_ = v_rot - h[:, None] * floor_plane['normal'][None, :]

        # main axis
        cov = np.cov(v_rot_, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov)
        v1 = eig_vec[:, np.argmax(eig_val)]
        v2 = np.cross(v1, floor_plane['normal'])

        # main axis diff
        v1_diff = abs(np.dot(v1, target_info['v1']))
        v2_diff = abs(np.dot(v2, target_info['v2']))
        iou = v1_diff + v2_diff
        # if iou < 1.5:
        #     continue

        v1_dist = np.dot(v_rot_, v1)
        v2_dist = np.dot(v_rot_, v2)
        v1_min, v1_max = v1_dist.min(), v1_dist.max()
        v2_min, v2_max = v2_dist.min(), v2_dist.max()
        
        sizes = np.array([
            v1_max - v1_min,
            v2_max - v2_min,
            h_max - h_min
        ])

        scale = (target_info['sizes'][:3] / sizes[:3])

        # each axis using different scale
        # unrotate to original axis

        scale_xy = 1/2 * (scale[0] + scale[1])

        # v_rot = np.array([
        #     v_rot[:, 0] * scale_xy,
        #     v_rot[:, 1] * scale_xy,
        #     v_rot[:, 2] * scale[2]
        # ]).T
        v_rot = scale_xy * (v1_dist[:,None] * v1 + v2_dist[:,None] * v2) + h[:,None] * floor_plane['normal'] * scale[2]
        # why this will skew the object
        # v_rot = np.dot(v_rot, rot_mat[:3, :3].T)
        
        # v_rot = v_rot * scale 

        # move to the floor
        height_diff = target_info['h_min'] - h_min * scale[2]
        shift = target_info['centroid'] + height_diff * floor_plane['normal']

        v_rot = v_rot + shift
        cad_mesh.vertices = v_rot

        #save v_rot to ply
        cad_mesh.export(result_dir / f'{group_id}_{angle}_rot.ply')
        


        # closest distance
        # _, distance, _ = trimesh.proximity.closest_point(mesh, target_info['pts'])
        # distance, _, _ = igl.point_mesh_squared_distance(target_info['pts'], v_rot, cad_mesh.faces)
        # compute point to point distance
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(v_rot)
        distances, _ = nbrs.kneighbors(target_info['pts'])
        pts_dist = np.mean(distances)


        tree_target = cKDTree(target_info['pts'])
        dist_v_rot = tree_target.query(v_rot, k=1)[0]

        pts_dist = pts_dist + dist_v_rot.mean()
        avg_dist = pts_dist + (2-iou) * iou_weight
        avg_dist = pts_dist 
        print('Angle:', angle, 'Distance:', avg_dist, 'IOU:', iou, 'dist_v_rot:', dist_v_rot.mean())
        if min_angle_dist is None or avg_dist < min_angle_dist:
            min_angle_dist = avg_dist
            best_angle = angle
            best_v = v_rot
            best_rot_mat = rot_mat  # best rotation matrix 저장
        
    print('Best Angle:', best_angle, 'Min Distance:', min_angle_dist)
    cad_mesh.vertices = best_v
    # best_rot_mat이 4x4라면, 상위 3x3만 추출
    if best_rot_mat is not None:
        quat = quaternion_from_matrix(best_rot_mat)
    else:
        quat = None
    # centroid는 target_info['centroid']
    pose_info = {
        'centroid': target_info['centroid'].tolist(),
        'quaternion': quat.tolist() if quat is not None else None
    }
    # pose_info를 result_dir/object_pose_tmp.json에 저장 (append 모드)
    pose_path = result_dir / 'object_poses.json'
    if pose_path.exists():
        with open(pose_path, 'r') as f:
            pose_list = json.load(f)
    else:
        pose_list = []
    pose_list.append(pose_info)
    with open(pose_path, 'w') as f:
        json.dump(pose_list, f, indent=2)
    return cad_mesh



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Object Fitting')
    parser.add_argument('--scene', type=str, default='data/room_716_refine')
    parser.add_argument('--try_num', type=int, default=20)
    args = parser.parse_args()

    scene_dir = Path(args.scene)
    result_root = Path('results')
    result_dir = result_root / scene_dir.stem
    result_dir.mkdir(exist_ok=True, parents=True)

    # target_categories = {
    #     'chair': {
    #         'shapenet_name': None,
    #         'label_id_in_scene': 5
    #     },
    #     'table': {
    #         'shapenet_name': None, 
    #         'label_id_in_scene': 7
    #     },
    #     'desk': {
    #         'shapenet_name': None, 
    #         'label_id_in_scene': 14
    #     },
    #     'shelf': {
    #         'shapenet_name': None,
    #         'label_id_in_scene': 10
    #     },
    #     'sofa': {
    #         'shapenet_name': None,
    #         'label_id_in_scene': 6
    #     },
    #     'cabinet': {
    #         'shapenet_name': None,
    #         'label_id_in_scene': 3
    #     }
    #     # 'door': {
    #     #     'shapenet_name': None,
    #     #     'label_id_in_scene': 8
    #     # }
    # }

    # target_label_ids = []
    # for k, v in target_categories.items():
    #     target_label_ids.append(v['label_id_in_scene'])

    # target_instances = load_data(scene_dir, target_label_ids, result_dir, conf_threshold)

    # taxonomy_path = 'G:/Dataset/ShapeNetCoreUnzip/shapenetcore.taxonomy.json'
    # target_categories = prepare_models(taxonomy_path, target_categories)

    # shapenet_dir = Path('G:/Dataset/ShapeNetCoreUnzip')
    # # all models
    # for k, v in target_categories.items():
    #     shapenet_path_list = list((shapenet_dir / v['shapenet_name']).glob('**/models/*.obj'))
    #     target_categories[k]['shapenet_path_list'] = shapenet_path_list
    #     print('Category:', k, 'Shapenet Name:', v['shapenet_name'], 'Model Num:', len(shapenet_path_list))


    # load ply
    ply_path = "/home/hazel/Downloads/" + "table.ply"
    mesh = trimesh.load(ply_path, maintain_order=True, process=False)
    # flip the axis y
    mesh.vertices[:, 1] = -mesh.vertices[:, 1]

    plane_path = "/home/hazel/Sources/SplaTAM/experiments/iPhone_Captures/241108113716/SplaTAM_iPhone/eval/floor/" + "plane.json"
    floor_plane = json.load(open(plane_path, "r"))

    # load group file 
    group_json_path = "/home/hazel/Sources/SplaTAM/experiments/iPhone_Captures/241108113716/SplaTAM_iPhone/eval/floor/" + "group.json"
    group = json.load(open(group_json_path, "r"))
    group = {int(k): v for k, v in group.items()}


    group = {3:6} # for testing
    group_set = set(group.values())
    inverted_index = {}
    for k, v in group.items():
        inverted_index[v] = inverted_index.get(v, []) + [k]

    for group_id in group_set:
        mask_ids = inverted_index[group_id]
        mesh_target = None
        for mask_id in mask_ids:
            pad_path = "/home/hazel/Sources/SplaTAM/experiments/iPhone_Captures/241108113716/SplaTAM_iPhone/eval/object_seg_ply/" + f"{mask_id}.ply"
            # mesh_ = trimesh.load(ply_path, maintain_order=True, process=False)
            mesh_ = o3d.io.read_triangle_mesh(pad_path)
            if mesh_target is None:
                mesh_target = mesh_
            else:
                mesh_target = mesh_target + mesh_
            print("Group id", group_id,"Mask id",mask_id)

        mesh = trimesh.load(ply_path, maintain_order=True, process=False)
        mesh.vertices[:, 1] = -mesh.vertices[:, 1]
        plydata = PlyData.read(ply_path)


        results = replace_by_models(floor_plane, mesh, np.array(mesh_target.vertices), result_dir, iou_weight=1e-1)
        results.export(result_dir / f'result_{group_id}.ply')

        plydata.elements[0]["x"] = results.vertices[:, 0]
        plydata.elements[0]["y"] = results.vertices[:, 1]
        plydata.elements[0]["z"] = results.vertices[:, 2]
        plydata.write(result_dir / f'result_gs_{group_id}.ply')


