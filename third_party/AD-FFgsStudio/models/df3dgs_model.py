import os

import torch
import torch.nn as nn
import yaml
from einops import rearrange

from models.network import DepthNetwork,GaussianNetwork

class DF3DGSModel(torch.nn.Module):
    def __init__(self, cfg):
        super(DF3DGSModel, self).__init__()
        self.read_config(cfg)
        self.depth_net = DepthNetwork(cfg)
        self.gaussian_net = GaussianNetwork(rgb_dim=3, depth_dim=1)
        self.novel_view_mode = self.depth_net.novel_view_mode


    def read_config(self, cfg):   
        for k, v in cfg.items():
            setattr(self, k, v)

    def load_official_weights(self,device='cuda:0'):

        weights_root = os.environ.get('DF3DGS_WEIGHTS_ROOT', f'weights_{self.novel_view_mode}')
        depth_weight_path = os.path.join(weights_root, 'depth_net.pth')
        gaussian_weight_path = os.path.join(weights_root, 'gs_net.pth')

        depth_weights = torch.load(depth_weight_path,map_location=device)

        depth_model_dict = self.depth_net.state_dict()
        # load parameters
        pre_trained_dict = {k: v for k, v in depth_weights.items() if k in depth_model_dict}
        depth_model_dict.update(pre_trained_dict)
        self.depth_net.load_state_dict(depth_model_dict)
        self.depth_net.eval()
        # for name, param in self.depth_net.named_parameters():
        #     print(name,param.shape, param.mean(), param.std())

        # self.depth_net.load_state_dict(depth_weights,strict=True)
        gaussian_weights = torch.load(gaussian_weight_path,map_location=device)
        gaussian_model_dict = self.gaussian_net.state_dict()
        # load parameters
        pre_trained_dict = {k: v for k, v in gaussian_weights.items() if k in gaussian_model_dict}
        gaussian_model_dict.update(pre_trained_dict)
        self.gaussian_net.load_state_dict(gaussian_model_dict)
        self.gaussian_net.eval()



    def forward(self, inputs, reconstruction_frames=[0]):
        """
        This function estimates the outputs of the network.
        """          
        # pre-calculate inverse of the extrinsic matrix
        # print(inputs.keys)
        # for key, val in inputs.items():
        #     if isinstance(val, torch.Tensor):
        #         print(key, val.shape)

        inputs['e2c_extr'] = torch.inverse(inputs['c2e_extr'])
        
        if self.novel_view_mode=="MF":
            assert len(reconstruction_frames) == 2
        elif self.novel_view_mode=="SF":
            assert reconstruction_frames == [0]
        else:
            raise NotImplementedError('bad novel_view_mode: {self.novel_view_mode}')
            
        # init dictionary 
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        depth_feats = self.depth_net(inputs,MF_frames=reconstruction_frames)
        
        depth_maps = []
        rot_maps = []
        scale_maps = []
        opacity_maps = []
        sh_maps = []
        for frame_id in reconstruction_frames:
            for cam in range(self.num_cams):
                image_data = inputs[('color_aug', frame_id)][:, cam, ...]

                image_feats = depth_feats[('cam', cam)][('img_feat', frame_id, 0)]

                ref_K = inputs[('K')][:, cam, ...]
                disp_data = depth_feats[('cam', cam)][('disp',frame_id, 0)]
                depth = self.to_depth(disp_data, ref_K)
                rot, scale, opacity, sh = self.gaussian_net(image_data, depth, image_feats)

                depth_maps.append(depth.squeeze(1))
                rot_maps.append(rot.permute(0,2,3,1))
                scale_maps.append(scale.permute(0,2,3,1))
                opacity_maps.append(opacity.squeeze(1))
                sh = rearrange(sh, 'b (h w) ... -> b h w ...',h=self.height,w=self.width)
                sh_maps.append(sh.squeeze(4).squeeze(3))


        depth_maps = torch.stack(depth_maps, dim=1).unsqueeze(-1)
        rot_maps = torch.stack(rot_maps, dim=1)
        scale_maps = torch.stack(scale_maps, dim=1)
        opacity_maps = torch.stack(opacity_maps, dim=1).unsqueeze(-1)
        sh_maps = torch.stack(sh_maps, dim=1)

        return depth_maps, rot_maps, scale_maps, opacity_maps, sh_maps

    def to_depth(self, disp_in, K_in):        
        """
        This function transforms disparity value into depth map while multiplying the value with the focal length.
        """
        min_disp = 1/self.max_depth
        max_disp = 1/self.min_depth
        disp_range = max_disp-min_disp

        disp_in = nn.functional.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1/disp

        return depth * K_in[:, 0:1, 0:1].unsqueeze(2)/self.focal_length_scale
    

if __name__ == '__main__':

    from dataset.vggt3dgs_data_module import VGGT3DGS_LITDataModule
    # dataset_args = {
    #     'cameras': cfg['data']['cameras'],
    #     'back_context': cfg['data']['back_context'],
    #     'forward_context': cfg['data']['forward_context'],
    #     'data_transform': get_transforms('train', **kwargs),
    #     'depth_type': cfg['data']['depth_type'] if 'gt_depth' in cfg['data']['train_requirements'] else None,
    #     'with_pose': 'gt_pose' in cfg['data']['train_requirements'],
    #     'with_ego_pose': 'gt_ego_pose' in cfg['data']['train_requirements'],
    #     'with_mask': 'mask' in cfg['data']['train_requirements']
    #     }
    '''
    dict_keys(['idx', 'token', 'sensor_name', 'filename', 'depth', 'extrinsics', 'mask', ('K', 0), ('inv_K', 0), ('color', 0, 0), ('color_aug', 0, 0), ('K', 1), ('inv_K', 1), ('color', 0, 1), ('color_aug', 0, 1), ('K', 2), ('inv_K', 2), ('color', 0, 2), ('color_aug', 0, 2), ('K', 3), ('inv
    _K', 3), ('color', 0, 3), ('color_aug', 0, 3), ('color', -1, 0), ('color_aug', -1, 0), ('cam_T_cam', 0, -1), ('color', 1, 0), ('color_aug', 1, 0), ('cam_T_cam', 0, 1)])
    '''

    cfg_path = 'configs/nuscenes/df3dgs.yaml'
    with open(cfg_path,'r') as fr:
        cfg = yaml.load(fr, Loader=yaml.FullLoader)

    model = DF3DGSModel(cfg['model_cfg'])
    model.load_official_weights()
    model.to('cuda:2')

    # data_module = VGGT3DGS_LITDataModule(cfg['data_cfg'])
    # data_module.setup()
    # train_dataloader = data_module.train_dataloader()

    import pickle
    with open('/home/kuntaoxiao/projects/DrivingForward/inputs.pkl', 'rb') as f:
        batch_inputs = pickle.load(f)
    sf_images = torch.stack([batch_inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(6)], 1) 
    print('sf_images',sf_images.shape,sf_images.mean(),sf_images.std())
    for key, val in batch_inputs.items():
        if isinstance(val, torch.Tensor):
            # batch_inputs
            batch_inputs[key] = val.float().to('cuda:2')
            print(key, val.shape,val.dtype,val.device)

    batch_inputs['c2e_extr'] = batch_inputs['extrinsics']
    batch_inputs['K'] = batch_inputs[('K',0)]
    for frame_id in [-1,0,1]:
        batch_inputs[('color_aug', frame_id)] = batch_inputs[('color', frame_id,0)]

    with torch.no_grad():
        outputs = model(batch_inputs)


