import yaml
import argparse
import os
import sys
import subprocess
import torch
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from utils.train_callback import ExportBestModelCallback, ExportMetricCallback
from dataset.vggt3dgs_data_module import VGGT3DGS_LITDataModule
from dataset.vggt4dgs_data_module import VGGT4DGS_LITDataModule
from models.vggt3dgs_model_module import VGGT3DGS_LITModelModule
from models.vggt4dgs_model_module import VGGT4DGS_LITModelModule


from utils.snapshot import save_pipeline_snapshot, PIPELINE_DEPLOYMENT

torch.set_float32_matmul_precision('highest')


# ==============================
# 新增：配置文件加载与合并函数
# ==============================
def load_and_merge_configs(main_cfg_path):
    """加载并合并主配置与子配置"""
    # 加载主配置文件
    with open(main_cfg_path) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)



    return main_cfg




def main():
    # 解析配置和路径参数
    parser = argparse.ArgumentParser(description='eval argparse')
    parser.add_argument('--cfg_path', type=str, required=True, help='主配置文件路径')
    # parser.add_argument('--only_eval', action='store_true', default=False)
    parser.add_argument('--restore_ckpt', type=str, default='')
    parser.add_argument('--train_4d', action='store_true', help='4dgs')
    args = parser.parse_args()

    # 加载并合并配置
    with open(args.cfg_path) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)

    main_cfg['model_cfg']['batch_size'] = main_cfg['data_cfg']['batch_size']

    save_dir = main_cfg['save_dir']


    # 创建子目录结构
    log_dir = os.path.join(save_dir, 'log')
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    code_dir = os.path.join(save_dir, 'code')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    save_pipeline_snapshot(PIPELINE_DEPLOYMENT, code_dir)
    with open(os.path.join(save_dir,'cfg.yaml'),'w') as fw:
        yaml.dump(main_cfg, fw)

    pl.seed_everything(main_cfg['seed'], workers=True)

    # 创建TensorBoard日志记录器
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name='logs'
    )

    # 初始化数据模块 - 使用配置文件参数
    if args.train_4d:
        data_module = VGGT4DGS_LITDataModule(
            cfg=main_cfg['data_cfg'],
        )
    else:
        data_module =  VGGT3DGS_LITDataModule(
            cfg=main_cfg['data_cfg'],
        )

    debug_dataset = data_module.construct_dataset('train')
    
    for idx in range(len(debug_dataset)):
        print('get debug data: ', idx)
        sample = debug_dataset[idx]
        breakpoint()

if __name__ == "__main__":
    main()