import glob
import os
import pickle
import time
import tqdm

import numpy as np
from PIL import Image
import torch
from einops import rearrange, repeat
from torch.utils.data import Dataset
from torchvision import transforms

from ..shims.augmentation_shim import apply_augmentation_shim


class WaymoDataset(Dataset):
    def __init__(
            self,
            mode,
            cfg,
    ):
        super().__init__()
        self.mode = mode
        self.data_path = cfg.data_path
        self.load_size = cfg.load_size
        self.num_cams = cfg.num_cams
        self.num_frames = cfg.num_frames
        
        self.context_ids = cfg.context_ids
        self.target_ids = cfg.target_ids
        
        assert self.num_cams==1
        
        self.ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]  # cam_id: 0 1 2 3 4
        
        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # parse the split file
        split_file = open(cfg.split_file, "r").readlines()[1:]
        scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
        start_timestep = [int(line.strip().split(",")[2]) for line in split_file]
        end_timestep = [int(line.strip().split(",")[3]) for line in split_file]
        
        eval_start_timestep = []
        eval_end_timestep = []
        for test_scene in cfg.test_scene_ids_list:
            id = scene_ids_list.index(test_scene)
            
            if test_scene in cfg.eval_scene_id:
                eval_start_timestep.append(start_timestep[id])
                eval_end_timestep.append(end_timestep[id])
            
            del scene_ids_list[id]
            del start_timestep[id]
            del end_timestep[id]
        
        scene_ids_list = cfg.eval_scene_id if self.mode!="train" else scene_ids_list
        start_timestep = eval_start_timestep if self.mode!="train" else start_timestep
        end_timestep = eval_end_timestep if self.mode!="train" else end_timestep
        self.all_datas = []
        for l_id, scene_id in enumerate(scene_ids_list):
            img_filepaths, lidar_depth_filepaths = self.create_all_filelist(scene_id, start_timestep[l_id], end_timestep[l_id])
            
            # creat frame
            # step = 1 if mode=="train" else cfg.num_frames
            step = {
                "train": 1,
                "val": 6,
                "test": cfg.num_frames
            }
            for frame_id in range(0, len(img_filepaths)//cfg.num_cams-cfg.num_frames, step[mode]):
                
                self.all_datas.append(
                    {
                        "scene_id": scene_id,
                        "img_filepaths": img_filepaths[frame_id:frame_id+cfg.num_frames*cfg.num_cams],
                        "lidar_depth_filepaths": lidar_depth_filepaths[frame_id:frame_id+cfg.num_frames*cfg.num_cams]
                    }
                )

    def create_all_filelist(self, scene_id, start_timestep, end_timestep):
        """
        Create file lists for all data files.
        e.g., img files, feature files, etc.
        """
        # ---- define camera list ---- #
        # 0: front, 1: front_left, 2: front_right, 3: side_left, 4: side_right
        if self.num_cams == 1:
            self.camera_list = [0]
        elif self.num_cams == 3:
            self.camera_list = [1, 0, 2]
        elif self.num_cams == 5:
            self.camera_list = [3, 1, 0, 2, 4]
        else:
            raise NotImplementedError(
                f"num_cams: {self.num_cams} not supported for waymo dataset"
            )

        # ---- define filepaths ---- #
        img_filepaths = []
        intrin_filepaths, extrin_filepaths, egop_filepaths = [], [], []
        lidar_depth_filepaths = []
        
        if end_timestep == -1:
            all_filepaths = os.path.join(self.data_path, str(scene_id).zfill(3), 'images', "*.png")
            image_filenames_all = glob.glob(all_filepaths)
            num_frames_all = len(image_filenames_all) // 5
            end_timestep = num_frames_all - 1

        # Note: we assume all the files in waymo dataset are synchronized
        for t in range(start_timestep, end_timestep):
            for cam_idx in self.camera_list:
                img_filepaths.append(
                    os.path.join(self.data_path, str(scene_id).zfill(3), "images", f"{t:06d}_{cam_idx}.png")
                )
                lidar_depth_filepaths.append(
                    os.path.join(self.data_path, str(scene_id).zfill(3), "lidar_depth", f"{t:06d}_{cam_idx}.npy")
                )
        
        return img_filepaths, lidar_depth_filepaths

    def load_calibrations(self, img_filepaths):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, ego-to-world matrices, etc.
        """
        # extract calib path and timestep
        start_filepath, end_filepath = img_filepaths[0], img_filepaths[-1]
        data_path, start_filename = os.path.split(start_filepath)
        _, end_filename = os.path.split(end_filepath)
        start_timestep = int(start_filename.split('.')[0][:6])
        end_timestep = int(end_filename.split('.')[0][:6])
        
        # to store per-camera intrinsics and extrinsics
        _intrinsics = []
        cam_to_egos = []
        for i in range(self.num_cams):
            # load camera intrinsics
            # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}].
            # ====!! we did not use distortion parameters for simplicity !!====
            # to be improved!!
            intrinsic = np.loadtxt(
                os.path.join(data_path.replace("images", "intrinsics"), f"{i}.txt")
            )
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            # normalize intrinsics w.r.t. ORIGINAL_SIZE
            fx, fy = (
                fx / self.ORIGINAL_SIZE[i][1],
                fy / self.ORIGINAL_SIZE[i][0],
            )
            cx, cy = (
                cx / self.ORIGINAL_SIZE[i][1],
                cy / self.ORIGINAL_SIZE[i][0],
            )
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            _intrinsics.append(intrinsic)

            # load camera extrinsics
            cam_to_ego = np.loadtxt(
                os.path.join(data_path.replace("images", "extrinsics"), f"{i}.txt")
            )
            # opencv coordinate system: x right, y down, z front
            cam_to_egos.append(cam_to_ego)

        # compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, cam_ids = [], []
        # ===! for waymo, we simplify timestamps as the time indices
        timestamps, timesteps = [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(
            os.path.join(data_path.replace("images", "ego_pose"), f"{start_timestep:06d}.txt")
        )
        for t in range(start_timestep, end_timestep+1):
            ego_to_world_current = np.loadtxt(
                os.path.join(data_path.replace("images", "ego_pose"), f"{t:06d}.txt")
            )
            # compute ego_to_world transformation
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            ego_to_worlds.append(ego_to_world)
            for cam_id in self.camera_list:
                cam_ids.append(cam_id)
                # transformation:
                #   (opencv_cam -> waymo_cam -> waymo_ego_vehicle) -> current_world
                ego_cam_pose = np.loadtxt(
                    os.path.join(data_path.replace("images", "ego_pose"), f"{t:06d}_{cam_id}.txt")
                )
                cam2world = ego_to_world @ cam_to_egos[cam_id]
                cam_to_worlds.append(torch.from_numpy(cam2world).float())
                intrinsics.append(torch.from_numpy(_intrinsics[cam_id]).float())
                # ===! we use time indices as the timestamp for waymo dataset for simplicity
                # ===! we can use the actual timestamps if needed
                # to be improved
                timestamps.append(t - start_timestep)
                timesteps.append(t - start_timestep)

        # intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        # cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        # ego_to_worlds = torch.from_numpy(np.stack(ego_to_worlds, axis=0)).float()
        # cam_ids = torch.from_numpy(np.stack(cam_ids, axis=0)).long()

        # # the underscore here is important.
        # _timestamps = torch.from_numpy(np.stack(timestamps, axis=0)).float()
        # _timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).long()
        
        return intrinsics, cam_to_worlds, ego_to_worlds

    def load_rgb(self, img_filepath):
        """
        Load the RGB images if they are available. We cache the images in memory for faster loading.
        Note this can be memory consuming.
        """
        rgb = Image.open(img_filepath).convert("RGB")
        # resize them to the load_size
        rgb = rgb.resize(
            (self.load_size[1], self.load_size[0]), Image.BILINEAR
        )
        # PIL to numpy
        rgb = np.array(rgb, dtype=np.float32, copy=False) / 255.0
        return rgb

    def load_lidar_depth(self, lidar_depth_filepath):
        """
        Load the lidar depths if they are available.
        """
        depth = np.load(lidar_depth_filepath, allow_pickle=True)
        depth = dict(depth.item())
        
        _, filename = os.path.split(lidar_depth_filepath)
        cam_id = int(filename.split('.')[0].split('_')[-1])
        scaling_h = self.ORIGINAL_SIZE[cam_id][0] / self.load_size[0]
        scaling_w = self.ORIGINAL_SIZE[cam_id][1] / self.load_size[1]
        
        u,v = depth['mask'].nonzero()
        u, v = u//int(scaling_h), v//int(scaling_w)
        depth_ = np.zeros(self.load_size).astype(np.float32)
        depth_[u, v] = depth['value']
            
        return depth_

    def __getitem__(self, index):
        scan = self.all_datas[index]
        
        img_filepaths = scan["img_filepaths"]
        lidar_depth_filepaths = scan["lidar_depth_filepaths"]
        
        images = []
        depths = []
        near = []
        far = []
        for i in range(len(img_filepaths)):
            rgb = self.load_rgb(img_filepaths[i])
            depth = self.load_lidar_depth(lidar_depth_filepaths[i])
        
            images.append(self.to_tensor(rgb))
            depths.append(torch.from_numpy(depth).float())
            
            near.append(torch.tensor(depth[depth!=0].min(), dtype=torch.float32))
            far.append(torch.tensor(depth[depth!=0].max(), dtype=torch.float32))
        
        intrinsics, cam_to_worlds, _ = self.load_calibrations(img_filepaths)
        
        data = {
            "context": {
                "extrinsics": torch.stack([cam_to_worlds[i] for i in self.context_ids]),
                "intrinsics": torch.stack([intrinsics[i] for i in self.context_ids]),
                "image": torch.stack([images[i] for i in self.context_ids]),
                "depth": torch.stack([depths[i] for i in self.context_ids]),
                "near": torch.stack([near[i] for i in self.context_ids]),
                "far": torch.stack([far[i] for i in self.context_ids]),
                "index": torch.from_numpy(np.arange(len(self.context_ids))),
            },
            "target": {
                "extrinsics": torch.stack([cam_to_worlds[i] for i in self.target_ids]),
                "intrinsics": torch.stack([intrinsics[i] for i in self.target_ids]),
                "image": torch.stack([images[i] for i in self.target_ids]),
                "depth": torch.stack([depths[i] for i in self.target_ids]),
                "near": torch.stack([near[i] for i in self.target_ids]),
                "far": torch.stack([far[i] for i in self.target_ids]),
                "index": torch.from_numpy(np.arange(len(self.target_ids))),
            },
            "scene": str(scan["scene_id"]),
        }
        
        if self.mode == "train":
            data = apply_augmentation_shim(data)
        
        return data

    def __len__(self):
        return len(self.all_datas)

