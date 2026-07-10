import os

import numpy as np
import PIL.Image as pil

import torch
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion

from dataset.data_util import img_loader, mask_loader_scene, align_3dgs_dataset, stack_sample

class NuScenesdataset(Dataset):
    """
    Loaders for NuScenes dataset
    """
    def __init__(self, path, stage,
                 cameras=None,
                 back_context=0,
                 forward_context=0,
                 data_transform=None,
                 with_pose=None,
                 with_mask=None,
                 version='interp_12Hz_trainval',
                 sample_hz=2,
                 ):        
        super().__init__()
        # version = 'v1.0-trainval'
        
        self.path = path
        self.cache_dir = os.environ.get('NUSCENES_CACHE_DIR', os.path.join(self.path, 'cache'))
        self.stage = stage
        self.dataset_idx = 0

        self.cameras = cameras
        self.num_cameras = len(cameras)
        self.sample_hz = sample_hz
        self.sample_rate = int(12/self.sample_hz)
        self.bwd = back_context
        self.fwd = forward_context
        if isinstance(self.bwd,int) and isinstance(self.fwd,int):
            if self.bwd==0 and self.fwd==0:
                self.has_context = False
            else:
                self.has_context = True
        else:
            raise ValueError(f'bwd({self.bwd}) and fwd({self.fwd}) Not supported')
        
        self.data_transform = data_transform

        self.with_pose = with_pose

        self.loader = img_loader

        self.with_mask = with_mask
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

        # 获取所有样本的token，并过滤出属于当前stage官方划分场景的样本
        self.sample_tokens = []
        for scene in self.dataset.scene:
            if scene['name'] in official_scene_names:  # 检查场景是否在当前stage的官方划分中
                sample_token = scene['first_sample_token']
                scene_sample_tokens = []
                while sample_token:
                    scene_sample_tokens.append(sample_token)
                    sample = self.dataset.get('sample', sample_token)
                    sample_token = sample['next']

                # 剔除前后无法获取完整context的帧
                # if self.has_context == 'by_num':
                #     scene_sample_tokens = scene_sample_tokens[self.bwd:len(scene_sample_tokens)-self.fwd]
                self.sample_tokens.extend(scene_sample_tokens)

        print('Num of samlpe_tokens: ',len(self.sample_tokens))

    def get_current(self, key, cam_sample):
        """
        This function returns samples for current contexts
        """        
        # get current timestamp rgb sample
        if key == 'rgb':
            rgb_path = cam_sample['filename']
            return self.loader(os.path.join(self.path, rgb_path))
        # get current timestamp camera intrinsics
        elif key == 'intrinsics':
            cam_param = self.dataset.get('calibrated_sensor', 
                                         cam_sample['calibrated_sensor_token'])
            return np.array(cam_param['camera_intrinsic'], dtype=np.float32)
        # get current timestamp camera extrinsics
        elif key == 'extrinsics':
            cam_param = self.dataset.get('calibrated_sensor', 
                                         cam_sample['calibrated_sensor_token'])
            return self.get_tranformation_mat(cam_param)
        else:
            raise ValueError('Unknown key: ' +key)

    def get_rgb_context_by_num(self, sample_token, cam_name):
        """
        This function returns samples for backward and forward contexts
        """

        # 现在，从这个关键帧图像开始，向前后遍历获取12Hz的完整序列
        # 首先向前遍历（找更早的图像）
        sample_nusc = self.dataset.get('sample', sample_token)
        cam_sample = self.dataset.get('sample_data', sample_nusc['data'][cam_name])

        prev_frame_ids = []
        prev_data_list = []
        prev_contexts= []
        prev_cam_trans = []
        prev_data = cam_sample
        prev_data_token = sample_nusc['prev']
        prev_frame_id = -1
        cam_T_cam = np.eye(4)
        for _ in range(self.bwd):
            if prev_data_token != '':
                prev_nusc = self.dataset.get('sample', prev_data_token)
                prev_data = self.dataset.get('sample_data', prev_nusc['data'][cam_name])
                prev_data_token = prev_nusc['prev']
                cam_T_cam = self.get_cam_T_cam(cam_sample, prev_data)

            prev_data_list.append(prev_data)
            prev_frame_ids.append(prev_frame_id)
            prev_contexts.append(self.get_current('rgb', prev_data))
            prev_cam_trans.append(cam_T_cam)
            prev_frame_id = prev_frame_id - 1
        
        
        # 然后向后遍历（找更晚的图像）
        next_frame_ids = []
        next_contexts = []
        next_data_list = []
        next_cam_trans = []
        next_data = cam_sample
        next_data_token = sample_nusc['next']
        next_frame_id = 1
        cam_T_cam = np.eye(4)
        for _ in range(self.fwd):
            if next_data_token != '':
                next_nusc = self.dataset.get('sample', next_data_token)
                next_data = self.dataset.get('sample_data', next_nusc['data'][cam_name])
                next_data_token = next_nusc['next']
                cam_T_cam = self.get_cam_T_cam(cam_sample, next_data)

                
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


    def get_tranformation_mat(self, pose):
        """
        This function transforms pose information in accordance with DDAD dataset format
        """
        extrinsics = Quaternion(pose['rotation']).transformation_matrix
        extrinsics[:3, 3] = np.array(pose['translation'])
        return extrinsics.astype(np.float32)

    def __len__(self):
        return len(self.sample_tokens) // self.sample_rate
    
    def __getitem__(self, idx):
        # get nuscenes dataset sample

        sample_idx = idx * self.sample_rate

        sample_token = self.sample_tokens[sample_idx]

        sample_nusc = self.dataset.get('sample', sample_token)
        scene_token = sample_nusc['scene_token']
        
        sample = []

        # loop over all cameras            
        for cam in self.cameras:
            cam_sample = self.dataset.get(
                'sample_data', sample_nusc['data'][cam])

            data = {
                'sample_token': sample_token,
                'scene_token': scene_token,
                ('filename',0): cam_sample['filename'],
                'rgb': self.get_current('rgb', cam_sample),
            }

            # if pose is returned
            if self.with_pose:
                data.update({
                    'extrinsics':self.get_current('extrinsics', cam_sample)
                })
                data.update({
                    'intrinsics': self.get_current('intrinsics', cam_sample)
                })

            # if mask is returned
            if self.with_mask:
                data.update({
                    'mask': self.mask_loader(self.mask_path, '', cam)
                })       

            data.update({('cam_T_cam', 0, 0): np.eye(4)})
            if self.has_context:
                context_frames, context_datas, context_rgbs, context_cam_trans = self.get_rgb_context_by_num(sample_token,cam)
            else:
                context_frames, context_datas, context_rgbs, context_cam_trans = [], [], [], []

            data.update({'rgb_context': context_rgbs,'frame_context': context_frames})
            for frame_id, frame_data, frame_cam_trans in zip(context_frames, context_datas, context_cam_trans):
                data.update({
                    ('filename',frame_id): frame_data['filename'],
                    ('cam_T_cam',0,frame_id): frame_cam_trans,
                })

            sample.append(data)

        if self.data_transform:
            sample = [self.data_transform(smp,) for smp in sample]

        # stack and align dataset for our trainer
        sample = stack_sample(sample)

        sample = align_3dgs_dataset(sample)
        return sample


if __name__ == '__main__':
    import yaml
    from functools import partial
    from dataset.data_util import train_transforms
    cfg_path = 'configs/nuscenes/vggt3dgs.yaml'
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
        'with_mask': 'mask' in data_cfg['train_requirements'],
        'sample_hz': 2,
    }

    test_dataset = NuScenesdataset(
        data_cfg['data_path'], 'test',
        **dataset_args   
    )

    print(len(test_dataset))
    test_frame = 10

    save_dir = 'work_dirs/dataset_test'
    test_data = test_dataset[test_frame]
    print(test_data.keys())

    for key, val in test_data.items():
        if isinstance(val, torch.Tensor):
            print(key, val.shape)
        else:
            print(key, val)

    frame_id = 0
    test_rgb0 = test_data[('color_aug',frame_id)][0].cpu().numpy().transpose(1,2,0)*255
    test_rgb0 = test_rgb0.astype(np.uint8)
    pil.fromarray(test_rgb0).save(os.path.join(save_dir,f'test_rgb{frame_id}.jpg'))
    for frame_id in test_data['frame_context']:
        test_rgb0 = test_data[('color_aug',frame_id)][0].cpu().numpy().transpose(1,2,0)*255
        test_rgb0 = test_rgb0.astype(np.uint8)
        test_cam_trans = test_data[('cam_T_cam',0,frame_id)].cpu().numpy()
        # print(frame_id,test_cam_trans)
        pil.fromarray(test_rgb0).save(os.path.join(save_dir,f'test_rgb{frame_id}.jpg'))
    