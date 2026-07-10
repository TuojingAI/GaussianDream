import os
import bisect
import numpy as np
import PIL.Image as pil
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion
from dataset.data_util import img_loader, mask_loader_scene, align_3dgs_dataset, stack_sample


class NuScenesSceneDataset(Dataset):
    """
    Scene-based NuScenes dataset loader that groups samples by scene
    Allows external iteration over scenes and access to all samples within a scene
    """
    
    def __init__(self, path, stage,
                 cameras=None,
                 back_context=0,
                 forward_context=0,
                 data_transform=None,
                 depth_type=None,
                 with_pose=None,
                 with_ego_pose=None,
                 with_mask=None,
                 test_scenes=None,
                 ):        
        super().__init__()
        version = 'v1.0-trainval'
        self.path = path
        self.cache_dir = os.environ.get('NUSCENES_CACHE_DIR', os.path.join(self.path, 'cache'))
        self.stage = stage
        self.dataset_idx = 0

        self.cameras = cameras
        self.num_cameras = len(cameras)
        self.bwd = back_context
        self.fwd = forward_context
        
        if isinstance(self.bwd,int) and isinstance(self.fwd,int):
            if self.bwd==0 and self.fwd==0:
                self.has_context = False
            else:
                self.has_context = 'by_num'
        elif isinstance(self.bwd,str) or isinstance(self.fwd,str):
            self.has_context = 'by_keyframe'
        else:
            raise ValueError(f'bwd({self.bwd}) and fwd({self.fwd}) Not supported')
        
        self.data_transform = data_transform

        self.with_depth = depth_type is not None
        self.with_pose = with_pose
        self.with_ego_pose = with_ego_pose

        self.loader = img_loader

        self.with_mask = with_mask
        self.test_scenes = test_scenes
        cur_path = os.path.dirname(os.path.realpath(__file__))        
        self.mask_path = os.path.join(cur_path, 'nuscenes_mask')
        self.mask_loader = mask_loader_scene

        self.dataset = NuScenes(version=version, dataroot=self.path, verbose=True)

        # 获取官方划分的场景列表
        if stage == 'train':
            official_scene_names = splits.train  # 官方训练划分场景
        elif stage == 'val':
            official_scene_names = splits.val    # 官方验证划分场景
        ## Hardcode: only specific scenes for testing
        elif stage == 'test':
            official_scene_names = [
                'scene-0014', 'scene-0018', 'scene-0906', 'scene-0098',
                'scene-0100', 'scene-0103', 'scene-0270', 'scene-0271',
                'scene-0278', 'scene-0553', 'scene-0558', 
                'scene-0802', 'scene-0968',  'scene-1065',
            ]
        else:
            raise ValueError("stage should be 'train' / 'val'/ 'test' ")

        # Organize data by scene
        self.scenes_data = []
        self.scene_names = []
        self.scene_tokens = []
        
        for scene in self.dataset.scene:
            if scene['name'] in official_scene_names:
                scene_name = scene['name']
                scene_token = scene['token']
                
                # Get all sample tokens for this scene
                sample_token = scene['first_sample_token']
                scene_sample_tokens = []
                visited_tokens = set()  # Prevent infinite loops
                while sample_token:
                    if sample_token in visited_tokens:
                        print(f"Warning: Detected circular reference in scene {scene_name}, breaking loop")
                        break
                    visited_tokens.add(sample_token)
                    scene_sample_tokens.append(sample_token)
                    sample = self.dataset.get('sample', sample_token)
                    sample_token = sample['next']
                
                if len(scene_sample_tokens) > 0:  # Only add scenes with valid samples
                    self.scenes_data.append(scene_sample_tokens)
                    self.scene_names.append(scene_name)
                    self.scene_tokens.append(scene_token)
        
        print(f'Number of scenes: {len(self.scenes_data)}')
        print(f'Total samples: {sum(len(scene) for scene in self.scenes_data)}')
        
        # Precompute cumulative lengths for efficient global index lookup
        self.cumulative_lengths = [0]
        for scene_samples in self.scenes_data:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(scene_samples))
        
    def get_num_scenes(self):
        """Return the number of scenes"""
        return len(self.scenes_data)
    
    def get_scene_length(self, scene_idx):
        """Return the number of samples in a specific scene"""
        return len(self.scenes_data[scene_idx])
    
    def get_scene_name(self, scene_idx):
        """Return the name of a specific scene"""
        return self.scene_names[scene_idx]
    
    def get_scene_token(self, scene_idx):
        """Return the token of a specific scene"""
        return self.scene_tokens[scene_idx]

    def get_current(self, key, cam_sample):
        """
        This function returns samples for current contexts
        """        
        if key == 'rgb':
            rgb_path = cam_sample['filename']
            return self.loader(os.path.join(self.path, rgb_path))
        elif key == 'intrinsics':
            cam_param = self.dataset.get('calibrated_sensor', 
                                         cam_sample['calibrated_sensor_token'])
            return np.array(cam_param['camera_intrinsic'], dtype=np.float32)
        elif key == 'extrinsics':
            cam_param = self.dataset.get('calibrated_sensor', 
                                         cam_sample['calibrated_sensor_token'])
            return self.get_tranformation_mat(cam_param)
        else:
            raise ValueError('Unknown key: ' + key)

    def get_rgb_context_by_num(self, cam_sample):
        """
        This function returns samples for backward and forward contexts
        """

        # 现在，从这个关键帧图像开始，向前后遍历获取12Hz的完整序列
        # 首先向前遍历（找更早的图像）
        prev_frame_ids = []
        prev_data_list = []
        prev_contexts= []
        prev_cam_trans = []
        prev_data = cam_sample
        prev_data_token = cam_sample['prev']
        prev_frame_id = -1
        for _ in range(self.bwd):
            if prev_data_token != '':
                prev_data = self.dataset.get('sample_data', prev_data_token)
                prev_data_token = prev_data['prev']

            prev_data_list.append(prev_data)
            prev_frame_ids.append(prev_frame_id)
            prev_contexts.append(self.get_current('rgb', prev_data))
            prev_cam_trans.append(self.get_cam_T_cam(cam_sample, prev_data))
            prev_frame_id = prev_frame_id - 1
        
        
        # 然后向后遍历（找更晚的图像）
        next_frame_ids = []
        next_contexts = []
        next_data_list = []
        next_cam_trans = []
        next_data = cam_sample
        next_data_token = cam_sample['next']
        next_frame_id = 1
        for _ in range(self.fwd):
            if next_data_token != '':
                next_data = self.dataset.get('sample_data', next_data_token)
                next_data_token = next_data['next']
                cam_T_cam = self.get_cam_T_cam(cam_sample, next_data)
            else:
                cam_T_cam = np.eye(4)
            next_data_list.append(next_data)
            next_frame_ids.append(next_frame_id)
            next_contexts.append(self.get_current('rgb', next_data))
            next_cam_trans.append(cam_T_cam)
            next_frame_id = next_frame_id + 1
        
        context_frames = prev_frame_ids + next_frame_ids 
        context_data_list = prev_data_list + next_data_list
        context_rgb_list = prev_contexts + next_contexts
        context_cam_trans = prev_cam_trans + next_cam_trans

        return context_frames,context_data_list, context_rgb_list, context_cam_trans
    
    def get_rgb_context_by_keyframe(self, cam_sample):
        """
        This function returns samples for backward and forward contexts
        """

        # 现在，从这个关键帧图像开始，向前后遍历获取12Hz的完整序列
        # 首先向前遍历（找更早的图像）

        prev_frame_ids = []
        prev_data_list = []
        prev_contexts= []
        prev_cam_trans = []
        if self.bwd=='exclusive_previous_keyframe' or self.bwd=='include_previous_keyframe':
            prev_data_token = cam_sample['prev']

            prev_frame_id = -1
            while prev_data_token != '':
                prev_data = self.dataset.get('sample_data', prev_data_token)
                if prev_data['is_key_frame'] and self.bwd=='exclusive_previous_keyframe':
                    break
                prev_data_list.append(prev_data)
                prev_frame_ids.append(prev_frame_id)
                prev_contexts.append(self.get_current('rgb', prev_data))
                prev_cam_trans.append(self.get_cam_T_cam(cam_sample, prev_data))
                prev_data_token = prev_data['prev']
                prev_frame_id = prev_frame_id - 1
                if prev_data['is_key_frame']:
                    break
        
        # 然后向后遍历（找更晚的图像）
        next_frame_ids = []
        next_contexts = []
        next_data_list = []
        next_cam_trans = []
        if self.fwd=='exclusive_next_keyframe' or self.fwd=='include_next_keyframe':
            next_data_token = cam_sample['next']
            next_frame_id = 1
            while next_data_token != '':
                next_data = self.dataset.get('sample_data', next_data_token)
                if next_data['is_key_frame'] and self.fwd=='exclusive_next_keyframe':
                    break
                next_data_list.append(next_data)
                next_frame_ids.append(next_frame_id)
                next_contexts.append(self.get_current('rgb', next_data))
                next_cam_trans.append(self.get_cam_T_cam(cam_sample, next_data))
                next_data_token = next_data['next']
                next_frame_id = next_frame_id + 1
                if next_data['is_key_frame']:
                    break
        
        context_frames = prev_frame_ids + next_frame_ids 
        context_data_list = prev_data_list + next_data_list
        context_rgb_list = prev_contexts + next_contexts
        context_cam_trans = prev_cam_trans + next_cam_trans

        return context_frames,context_data_list, context_rgb_list, context_cam_trans

    def filter_rgb_context(self, sample_cams_list):
        # for sample in sample_cams_list:
        #     print('before',sample['frame_context'])
        len_camera = len(sample_cams_list)
        min_cam_id = 0
        min_context_len = len(sample_cams_list[min_cam_id]['frame_context'])
        for cam_id in range(1,len_camera):
            if len(sample_cams_list[cam_id]['frame_context']) < min_context_len:
                min_context_len = len(sample_cams_list[cam_id]['frame_context'])
                min_cam_id = cam_id

        index = 0
        while index < len(sample_cams_list[min_cam_id]['frame_context']):
            frame_id = sample_cams_list[min_cam_id]['frame_context'][index]
            for cam_id in range(len_camera):
                while frame_id != sample_cams_list[cam_id]['frame_context'][index]:
                    del_frame_id = sample_cams_list[cam_id]['frame_context'][index]
                    del sample_cams_list[cam_id][('filename',del_frame_id)]
                    del sample_cams_list[cam_id][('cam_T_cam',0,del_frame_id)]
                    del sample_cams_list[cam_id]['frame_context'][index]
                    del sample_cams_list[cam_id]['rgb_context'][index]

            index += 1
        
        for cam_id in range(len_camera):
            while len(sample_cams_list[cam_id]['frame_context'])>min_context_len:
                del_index = min_context_len
                del_frame_id = sample_cams_list[cam_id]['frame_context'][del_index]
                del sample_cams_list[cam_id][('filename',del_frame_id)]
                del sample_cams_list[cam_id][('cam_T_cam',0,del_frame_id)]    
                del sample_cams_list[cam_id]['frame_context'][del_index]
                del sample_cams_list[cam_id]['rgb_context'][del_index]
        # for sample in sample_cams_list:
        #     print('after',sample['frame_context'])
        return sample_cams_list
            

    def get_cam_T_cam(self, from_sample, to_sample):
        # from_sample -> world
        from_ego_pose = self.dataset.get('ego_pose', from_sample['ego_pose_token'])
        from_ego_rotation = Quaternion(from_ego_pose['rotation']).inverse
        from_ego_translation = -np.array(from_ego_pose['translation'])[:, None]
        world_to_from_ego = np.vstack([
            np.hstack((from_ego_rotation.rotation_matrix,
                        from_ego_rotation.rotation_matrix @ from_ego_translation)),
            np.array([0, 0, 0, 1])
        ])
        
        from_ego_to_world = np.linalg.inv(world_to_from_ego)
        
        from_cam_to_ego = self.dataset.get('calibrated_sensor', from_sample['calibrated_sensor_token'])
        from_cam_rotation = Quaternion(from_cam_to_ego['rotation'])
        from_cam_translation = np.array(from_cam_to_ego['translation'])[:, None]
        from_cam_to_ego_mat = np.vstack([
            np.hstack((from_cam_rotation.rotation_matrix, from_cam_translation)),
            np.array([0, 0, 0, 1])
        ])
        
        # to_sample -> world
        to_ego_pose = self.dataset.get('ego_pose', to_sample['ego_pose_token'])
        to_ego_rotation = Quaternion(to_ego_pose['rotation']).inverse
        to_ego_translation = -np.array(to_ego_pose['translation'])[:, None]
        world_to_to_ego = np.vstack([
            np.hstack((to_ego_rotation.rotation_matrix,
                        to_ego_rotation.rotation_matrix @ to_ego_translation)),
            np.array([0, 0, 0, 1])
        ])
        
        to_cam_to_ego = self.dataset.get('calibrated_sensor', to_sample['calibrated_sensor_token'])
        to_cam_rotation = Quaternion(to_cam_to_ego['rotation'])
        to_cam_translation = np.array(to_cam_to_ego['translation'])[:, None]
        to_cam_to_ego_mat = np.vstack([
            np.hstack((to_cam_rotation.rotation_matrix, to_cam_translation)),
            np.array([0, 0, 0, 1])
        ])
        to_ego_to_to_cam = np.linalg.inv(to_cam_to_ego_mat)
        
        # 正确的变换顺序应该是:
        # from_cam -> from_ego -> world -> to_ego -> to_cam
        cam_T_cam = to_ego_to_to_cam @ world_to_to_ego @ from_ego_to_world @ from_cam_to_ego_mat

        return cam_T_cam

    def generate_depth_map(self, sample, sensor, cam_sample):
        """
        This function returns depth map for nuscenes dataset,
        result of depth map is saved in nuscenes/samples/DEPTH_MAP
        """        
        filename = '{}/{}.npz'.format(
                        os.path.join(os.path.dirname(self.cache_dir), 'samples'),
                        'DEPTH_MAP/{}/{}'.format(sensor, cam_sample['filename']))
        
        load_flag = False
        if os.path.exists(filename):
            try:
                depth = np.load(filename, allow_pickle=True)['depth']
                load_flag = True
            except:
                load_flag = False
            
        if not load_flag:
            lidar_sample = self.dataset.get(
                'sample_data', sample['data']['LIDAR_TOP'])

            lidar_file = os.path.join(
                self.path, lidar_sample['filename'])
            lidar_points = np.fromfile(lidar_file, dtype=np.float32)
            lidar_points = lidar_points.reshape(-1, 5)[:, :3]

            lidar_pose = self.dataset.get(
                'ego_pose', lidar_sample['ego_pose_token'])
            lidar_rotation= Quaternion(lidar_pose['rotation'])
            lidar_translation = np.array(lidar_pose['translation'])[:, None]
            lidar_to_world = np.vstack([
                np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
                np.array([0, 0, 0, 1])
            ])

            sensor_sample = self.dataset.get(
                'calibrated_sensor', lidar_sample['calibrated_sensor_token'])
            lidar_to_ego_rotation = Quaternion(
                sensor_sample['rotation']).rotation_matrix
            lidar_to_ego_translation = np.array(
                sensor_sample['translation']).reshape(1, 3)

            ego_lidar_points = np.dot(
                lidar_points[:, :3], lidar_to_ego_rotation.T)
            ego_lidar_points += lidar_to_ego_translation

            homo_ego_lidar_points = np.concatenate(
                (ego_lidar_points, np.ones((ego_lidar_points.shape[0], 1))), axis=1)

            ego_pose = self.dataset.get(
                    'ego_pose', cam_sample['ego_pose_token'])
            ego_rotation = Quaternion(ego_pose['rotation']).inverse
            ego_translation = - np.array(ego_pose['translation'])[:, None]
            world_to_ego = np.vstack([
                    np.hstack((ego_rotation.rotation_matrix,
                               ego_rotation.rotation_matrix @ ego_translation)),
                    np.array([0, 0, 0, 1])
                    ])

            sensor_sample = self.dataset.get(
                'calibrated_sensor', cam_sample['calibrated_sensor_token'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(
                sensor_sample['translation'])[:, None]
            sensor_to_ego = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, 
                           sensor_translation)),
                np.array([0, 0, 0, 1])
               ])
            ego_to_sensor = np.linalg.inv(sensor_to_ego)
            
            lidar_to_sensor = ego_to_sensor @ world_to_ego @ lidar_to_world
            homo_ego_lidar_points = torch.from_numpy(homo_ego_lidar_points).float()
            cam_lidar_points = np.matmul(lidar_to_sensor, homo_ego_lidar_points.T).T

            depth_mask = cam_lidar_points[:, 2] > 0
            cam_lidar_points = cam_lidar_points[depth_mask]

            intrinsics = np.eye(4)
            intrinsics[:3, :3] = sensor_sample['camera_intrinsic']
            pixel_points = np.matmul(intrinsics, cam_lidar_points.T).T
            pixel_points[:, :2] /= pixel_points[:, 2:3]
            
            image_filename = os.path.join(
                self.path, cam_sample['filename'])
            img = pil.open(image_filename)
            h, w, _ = np.array(img).shape
            
            pixel_mask = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] <= w-1)\
                        & (pixel_points[:,1] >= 0) & (pixel_points[:,1] <= h-1)
            valid_points = pixel_points[pixel_mask].round().int()
            valid_depth = cam_lidar_points[:, 2][pixel_mask]
        
            depth = np.zeros([h, w])
            depth[valid_points[:, 1], valid_points[:,0]] = valid_depth
        
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, depth=depth)
        return depth

    def get_tranformation_mat(self, pose):
        """
        This function transforms pose information in accordance with DDAD dataset format
        """
        extrinsics = Quaternion(pose['rotation']).transformation_matrix
        extrinsics[:3, 3] = np.array(pose['translation'])
        return extrinsics.astype(np.float32)

    def get_scene_sample(self, scene_idx, sample_idx):
        """
        Get a specific sample from a specific scene
        Args:
            scene_idx: Index of the scene
            sample_idx: Index of the sample within the scene
        Returns:
            Processed sample data
        """
        if scene_idx >= len(self.scenes_data):
            raise IndexError(f"Scene index {scene_idx} out of range")
        if sample_idx >= len(self.scenes_data[scene_idx]):
            raise IndexError(f"Sample index {sample_idx} out of range for scene {scene_idx}")
        
        keyframe_token = self.scenes_data[scene_idx][sample_idx]
        sample_nusc = self.dataset.get('sample', keyframe_token)
        scene_token = sample_nusc['scene_token']

        sample = []

        # loop over all cameras            
        for cam in self.cameras:
            cam_sample = self.dataset.get(
                'sample_data', sample_nusc['data'][cam])

            data = {
                'idx': f'{scene_idx}_{sample_idx}',
                'sample_token': keyframe_token,
                'scene_token': scene_token,
                ('filename',0): cam_sample['filename'],
                'rgb': self.get_current('rgb', cam_sample),
                'intrinsics': self.get_current('intrinsics', cam_sample)
            }

            # if depth is returned            
            if self.with_depth:
                data.update({
                    'depth': self.generate_depth_map(sample_nusc, cam, cam_sample)
                })
            # if pose is returned
            if self.with_pose:
                data.update({
                    'extrinsics':self.get_current('extrinsics', cam_sample)
                })

            # if mask is returned
            if self.with_mask:
                data.update({
                    'mask': self.mask_loader(self.mask_path, '', cam)
                })        
            
            data.update({('cam_T_cam', 0, 0): np.eye(4)})
            # if context is returned
            if self.has_context=='by_num':
                context_frames, context_datas, context_rgbs, context_cam_trans = self.get_rgb_context_by_num(cam_sample)
            elif self.has_context=='by_keyframe':
                context_frames, context_datas, context_rgbs, context_cam_trans = self.get_rgb_context_by_keyframe(cam_sample)
            else:
                context_frames, context_datas, context_rgbs, context_cam_trans = [], [], [], []
            data.update({'rgb_context': context_rgbs,'frame_context': context_frames})

            for frame_id, frame_data, frame_cam_trans in zip(context_frames, context_datas, context_cam_trans):
                data.update({
                    ('filename',frame_id): frame_data['filename'],
                    ('cam_T_cam',0,frame_id): frame_cam_trans,
                })

            sample.append(data)

        if self.has_context=='by_keyframe':
            sample = self.filter_rgb_context(sample)

        # apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp,) for smp in sample]

        # stack and align dataset for our trainer
        sample = stack_sample(sample)
        sample = align_3dgs_dataset(sample)
        return sample

    def get_scene_samples(self, scene_idx):
        """
        Get all samples from a specific scene
        Args:
            scene_idx: Index of the scene
        Returns:
            List of processed sample data for the entire scene
        """
        scene_samples = []
        scene_length = self.get_scene_length(scene_idx)
        
        for sample_idx in range(scene_length):
            sample = self.get_scene_sample(scene_idx, sample_idx)
            scene_samples.append(sample)
        
        return scene_samples

    # Legacy methods for compatibility with standard Dataset interface
    def __len__(self):
        """Total number of samples across all scenes (for standard Dataset compatibility)"""
        return sum(len(scene) for scene in self.scenes_data)
    
    def __getitem__(self, idx):
        """Get sample by global index (for standard Dataset compatibility)"""
        # Use binary search for efficient O(log n) lookup
        if idx < 0 or idx >= self.cumulative_lengths[-1]:
            raise IndexError(f"Index {idx} out of range [0, {self.cumulative_lengths[-1]})")
        
        # Binary search to find the scene containing this global index
        scene_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        sample_idx = idx - self.cumulative_lengths[scene_idx]
        return self.get_scene_sample(scene_idx, sample_idx)
    
if __name__ == '__main__':
    import yaml
    from functools import partial
    from dataset.data_util import train_transforms
    cfg_path = 'configs/nuscenes/df3dgs_inference.yaml'
    with open(cfg_path) as f:
        data_cfg = yaml.load(f, Loader=yaml.FullLoader)['data_cfg']

    crop_scale = data_cfg.get('crop_scale',[])
    crop_ratio = data_cfg.get('crop_ratio',0.0)
    crop_prob = data_cfg.get('crop_prob',[])
    jittering = data_cfg.get('jittering',[])
    jittering_prob = data_cfg.get('jittering_prob',0.0)

    mode = 'test'
    dataset_args = {
        'cameras': data_cfg['cameras'],
        'back_context': data_cfg['back_context'],
        'forward_context': data_cfg['forward_context'],
        'data_transform': partial(train_transforms,
                    image_shape=(int(data_cfg['height']), int(data_cfg['width'])),
                    crop_scale=crop_scale if mode=='train' else [],
                    crop_ratio=crop_ratio if mode=='train' else [],
                    crop_prob=crop_prob if mode=='train' else 0.0,
                    jittering=jittering if mode=='train' else [],
                    jittering_prob=jittering_prob if mode=='train' else 0.0,),
        'depth_type': data_cfg['depth_type'] if 'gt_depth' in data_cfg['train_requirements'] else None,
        'with_pose': 'gt_pose' in data_cfg['train_requirements'],
        'with_ego_pose': 'gt_ego_pose' in data_cfg['train_requirements'],
        'with_mask': 'mask' in data_cfg['train_requirements']
    }

    test_dataset = NuScenesSceneDataset(
        data_cfg['data_path'], 'test',
        **dataset_args   
    )

    print(test_dataset.scene_names)
    test_scene_idx = 0
    test_frame = 10

    save_dir = 'work_dirs/dataset_test'

    test_data = test_dataset.get_scene_sample(scene_idx=test_scene_idx, sample_idx=test_frame)
    print(test_data.keys())
    # print(test_data['frame_context'])
    # for key, val in test_data.items():
    #     if isinstance(val, torch.Tensor):
    #         print(key, val.shape)
        # else:
        #     print(key, val)

    # frame_id = 0
    # test_rgb0 = test_data[('color_aug',frame_id)][0].cpu().numpy().transpose(1,2,0)*255
    # test_rgb0 = test_rgb0.astype(np.uint8)
    # pil.fromarray(test_rgb0).save(os.path.join(save_dir,f'test_rgb{frame_id}.jpg'))
    # for frame_id in test_data['frame_context']:
    #     test_rgb0 = test_data[('color_aug',frame_id)][0].cpu().numpy().transpose(1,2,0)*255
    #     test_rgb0 = test_rgb0.astype(np.uint8)
    #     test_cam_trans = test_data[('cam_T_cam',0,frame_id)].cpu().numpy()
    #     print(frame_id,test_cam_trans)
    #     pil.fromarray(test_rgb0).save(os.path.join(save_dir,f'test_rgb{frame_id}.jpg'))



    # print(test_dataset.sample_tokens[:3])
    # for key, val in test_dataset[1].items():
    #     if isinstance(val, torch.Tensor):
    #         print(key, val.shape)
    #     else:
    #         print(key, val)

    # test_rgb2 = test_dataset[1][('color_aug',0)][0].cpu().numpy().transpose(1,2,0)*255
    # test_rgb2 = test_rgb2.astype(np.uint8)
    # pil.fromarray(test_rgb2).save(os.path.join(save_dir,'test_rgb2.jpg'))
