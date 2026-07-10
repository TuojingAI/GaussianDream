import os
import glob
import os
from PIL import Image
from tqdm import tqdm
import torch
import sys
from torch import Tensor
from lpips import LPIPS
from einops import rearrange, reduce
from jaxtyping import Float, UInt8
from skimage.metrics import structural_similarity
import torchvision.transforms as transforms
import json
import numpy as np
import pandas as pd


class Metrics:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.lpips = LPIPS(net="vgg")
        self.lpips.eval().to(self.device)
        # 确保LPIPS的所有参数和缓冲区都在指定设备上
        for param in self.lpips.parameters():
            param.data = param.data.to(self.device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(self.device)
        for buffer in self.lpips.buffers():
            buffer.data = buffer.data.to(self.device)
        self.transform = transforms.ToTensor()

    @torch.no_grad()
    def compute_psnr(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ground_truth = ground_truth.clip(min=0, max=1)
        predicted = predicted.clip(min=0, max=1)
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
        return -10 * mse.log10()

    @torch.no_grad()
    def compute_lpips(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        # 确保输入张量在同一设备上
        ground_truth = ground_truth.to(self.device)
        predicted = predicted.to(self.device)
        value = self.lpips.forward(ground_truth, predicted, normalize=True)
        return value[:, 0, 0, 0]

    @torch.no_grad()
    def compute_ssim(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ssim = [
            structural_similarity(
                gt.detach().cpu().numpy(),
                hat.detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for gt, hat in zip(ground_truth, predicted)
        ]
        return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)

    def calculate_sample_new(self, sample_path):
        gt_imgs = glob.glob(os.path.join(sample_path, 'gt_views', '*_gt.png'))
        gt_psnr_list = []
        gt_ssim_list = []
        gt_lspips_list = []
        for gt_img_path in gt_imgs:
            gt_img = np.array(Image.open(gt_img_path))
            gt_img = self.transform(gt_img).unsqueeze(0)
            pred_img = np.array(Image.open(gt_img_path.replace('_gt.png', '_pred.png')))
            pred_img = self.transform(pred_img).unsqueeze(0)
            psnr_val = self.compute_psnr(gt_img, pred_img).mean().item()
            ssim_val = self.compute_ssim(gt_img, pred_img).mean().item()
            lpips_val = self.compute_lpips(gt_img, pred_img).mean().item()

            gt_psnr_list.append(psnr_val)
            gt_ssim_list.append(ssim_val)
            gt_lspips_list.append(lpips_val)

        return gt_psnr_list, gt_ssim_list, gt_lspips_list
    
    def calculate_sample_old(self, sample_path):
        gt_imgs = glob.glob(os.path.join(sample_path, 'gt_views', '*_gt.png'))
        gt_psnr_list = []
        gt_ssim_list = []
        gt_lspips_list = []
        for gt_img_path in gt_imgs:
            gt_img = np.array(Image.open(gt_img_path))
            gt_img = self.transform(gt_img).unsqueeze(0).to(self.device)
            pred_img = np.array(Image.open(gt_img_path.replace('_gt.png', '_pred.png')))
            pred_img = self.transform(pred_img).unsqueeze(0).to(self.device)
            psnr_val = self.compute_psnr(gt_img, pred_img).mean().item()
            ssim_val = self.compute_ssim(gt_img, pred_img).mean().item()
            lpips_val = self.compute_lpips(gt_img, pred_img).mean().item()

            gt_psnr_list.append(psnr_val)
            gt_ssim_list.append(ssim_val)
            gt_lspips_list.append(lpips_val)

        
        nv_psnr_list, nv_ssim_list, nv_lspips_list = [], [], []

        for nv_dir in ['timestep_1','timestep_2','timestep_3','timestep_4','timestep_5']:
            gt_imgs = glob.glob(os.path.join(sample_path, nv_dir, '*_gt.png'))
            for gt_img_path in gt_imgs:
                gt_img = np.array(Image.open(gt_img_path))
                gt_img = self.transform(gt_img).unsqueeze(0).to(self.device)
                pred_img = np.array(Image.open(gt_img_path.replace('_gt.png', '_pred.png')))
                pred_img = self.transform(pred_img).unsqueeze(0).to(self.device)
                psnr_val = self.compute_psnr(gt_img, pred_img).mean().item()
                ssim_val = self.compute_ssim(gt_img, pred_img).mean().item()
                lpips_val = self.compute_lpips(gt_img, pred_img).mean().item()

                nv_psnr_list.append(psnr_val)
                nv_ssim_list.append(ssim_val)
                nv_lspips_list.append(lpips_val)


        return gt_psnr_list, gt_ssim_list, gt_lspips_list, nv_psnr_list, nv_ssim_list, nv_lspips_list
            
sample_metrics = Metrics('cuda:1')        

def calculate_scene_old(scene_path):
    
    scene_gt_psnr_list, scene_gt_ssim_list, scene_gt_lspips_list, scene_nv_psnr_list, scene_nv_ssim_list, scene_nv_lspips_list = [], [], [], [], [], []
    for sample_name in os.listdir(scene_path):
        sample_id = int(sample_name.split('_')[-1])
        if sample_id < 228:
            gt_psnr_list, gt_ssim_list, gt_lspips_list, nv_psnr_list, nv_ssim_list, nv_lspips_list = sample_metrics.calculate_sample_old(os.path.join(scene_path, sample_name))
            print(f'==========={sample_name}===============')
            print(len(gt_psnr_list),len(nv_psnr_list))
            print(np.mean(gt_psnr_list), np.mean(gt_ssim_list), np.mean(gt_lspips_list))
            print(np.mean(nv_psnr_list), np.mean(nv_ssim_list), np.mean(nv_lspips_list))
            scene_gt_psnr_list.extend(gt_psnr_list)
            scene_gt_ssim_list.extend(gt_ssim_list)
            scene_gt_lspips_list.extend(gt_lspips_list)
            scene_nv_psnr_list.extend(nv_psnr_list)
            scene_nv_ssim_list.extend(nv_ssim_list)
            scene_nv_lspips_list.extend(nv_lspips_list)
    
    return scene_gt_ssim_list, scene_gt_ssim_list, scene_gt_lspips_list, scene_nv_psnr_list, scene_nv_ssim_list, scene_nv_lspips_list

def calculate_scene_new(scene_path):
    
    scene_gt_psnr_list, scene_gt_ssim_list, scene_gt_lspips_list, scene_nv_psnr_list, scene_nv_ssim_list, scene_nv_lspips_list = [], [], [], [], [], []
    for sample_name in tqdm(os.listdir(scene_path)):
        sample_id = int(sample_name.split('_')[-1])
        if sample_id < 228:
            gt_psnr_list, gt_ssim_list, gt_lspips_list = sample_metrics.calculate_sample_new(os.path.join(scene_path, sample_name))

            if sample_id % 6 == 0:
                scene_gt_psnr_list.extend(gt_psnr_list)
                scene_gt_ssim_list.extend(gt_ssim_list)
                scene_gt_lspips_list.extend(gt_lspips_list)
            else:
                scene_nv_psnr_list.extend(gt_psnr_list)
                scene_nv_ssim_list.extend(gt_ssim_list)
                scene_nv_lspips_list.extend(gt_lspips_list)
    
    return scene_gt_psnr_list, scene_gt_ssim_list, scene_gt_lspips_list, scene_nv_psnr_list, scene_nv_ssim_list, scene_nv_lspips_list



if __name__ == '__main__':
    root_path = './work_dirs/vggt4dgs_1221_12hz'
    # rename_folders(root_path)
    
    all_gt_psnr_list = []
    all_gt_ssim_list = []
    all_gt_lspips_list = []
    all_nv_psnr_list = []
    all_nv_ssim_list = []
    all_nv_lspips_list = []
    scene_list = os.listdir(root_path)

    result_dict = {}
    for scene_name in tqdm(scene_list):
        if os.path.isdir(os.path.join(root_path, scene_name)):
            scene_path = os.path.join(root_path, scene_name)
            scene_gt_psnr_list, scene_gt_ssim_list, scene_gt_lspips_list, scene_nv_psnr_list, scene_nv_ssim_list, scene_nv_lspips_list = calculate_scene_new(scene_path)
            
            print(f'==========={scene_name}===============')
            print(len(scene_gt_psnr_list),len(scene_nv_psnr_list))
            print(np.mean(scene_gt_psnr_list), np.mean(scene_gt_ssim_list), np.mean(scene_gt_lspips_list))
            print(np.mean(scene_nv_psnr_list), np.mean(scene_nv_ssim_list), np.mean(scene_nv_lspips_list))
            result_dict[scene_name] = {
                'gt_psnr': scene_gt_psnr_list,
                'gt_ssim': scene_gt_ssim_list,
                'gt_lspips': scene_gt_lspips_list,
                'nv_psnr': scene_nv_psnr_list,
                'nv_ssim': scene_nv_ssim_list,
                'nv_lspips': scene_nv_lspips_list
            }
            all_gt_psnr_list.extend(scene_gt_psnr_list)
            all_gt_ssim_list.extend(scene_gt_ssim_list)
            all_gt_lspips_list.extend(scene_gt_lspips_list)
            all_nv_psnr_list.extend(scene_nv_psnr_list)
            all_nv_ssim_list.extend(scene_nv_ssim_list)
            all_nv_lspips_list.extend(scene_nv_lspips_list)
    

    final_recon_psnr = np.mean(all_gt_psnr_list)
    final_recon_ssim = np.mean(all_gt_ssim_list)
    final_recon_lpips = np.mean(all_gt_lspips_list)
    final_novel_psnr = np.mean(all_nv_psnr_list)
    final_novel_ssim = np.mean(all_nv_ssim_list)
    final_novel_lpips = np.mean(all_nv_lspips_list)

    print('Overall')
    print(f"\nScene Reconstruction (Frame 0):")
    print(f"  PSNR: {final_recon_psnr:.4f}, SSIM: {final_recon_ssim:.4f}, LPIPS: {final_recon_lpips:.4f}")
    print(f"\nNovel View Synthesis (Middle Frames):")
    print(f"  PSNR: {final_novel_psnr:.4f}, SSIM: {final_novel_ssim:.4f}, LPIPS: {final_novel_lpips:.4f}")

    result_path = os.path.join(root_path, 'results.json')
    with open(result_path, 'w') as f:
        json.dump(result_dict, f)

    # uniad_path = '/workspace/data_storage/ruiyanghao/projects/uniadv2/UniAD'
    # sys.path.append(uniad_path)
    # from uniad_api import run_uniad

    # eval_sets = [ 
    #     ("normal_gen", 0),
    #     ("left", 1), ("left", 2), ("left", 3),
    #     ("right", 1), ("right", 2), ("right", 3)
    # ]

    # all_metrics = {}

    # for eval_set in eval_sets:
    #     print("Eval Setting:", eval_set)
    #     metrics = run_uniad(
    #         generated_path=root_path,
    #         translation_mode=eval_set[0],
    #         translation_offset=eval_set[1]
    #     )
    #     print(metrics)

    #     # 用字符串作为 key，更直观
    #     key = f"{eval_set[0]}_{eval_set[1]}"
    #     all_metrics[key] = metrics

    # df = pd.DataFrame.from_dict(all_metrics, orient="index")

    # mean_metrics = df.mean().round(3)

    # df.loc["mean"] = mean_metrics

    # print("\n==== Summary of All Metrics ====")
    # print(df)
