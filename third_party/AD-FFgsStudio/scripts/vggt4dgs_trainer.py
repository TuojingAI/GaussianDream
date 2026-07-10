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

from dataset.vggt4dgs_data_module import VGGT4DGS_LITDataModule
from models.vggt4dgs_model_module import VGGT4DGS_LITModelModule


from utils.snapshot import save_pipeline_snapshot

torch.set_float32_matmul_precision('high')

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
    save_pipeline_snapshot(code_dir)
    with open(os.path.join(save_dir,'cfg.yaml'),'w') as fw:
        yaml.dump(main_cfg, fw)

    pl.seed_everything(main_cfg['seed'], workers=True)

    # 创建TensorBoard日志记录器
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name='logs'
    )

    data_module =  VGGT4DGS_LITDataModule(
        cfg=main_cfg['data_cfg'],
    )


    litmodel = VGGT4DGS_LITModelModule(
            cfg=main_cfg['model_cfg'],
            save_dir=log_dir,
            logger=logger
        )

    if args.restore_ckpt:
        load_ckpt = torch.load(args.restore_ckpt,map_location="cpu")
        litmodel.load_state_dict(load_ckpt['state_dict'],strict=False)

    # 删除重复的检查点回调（已在模型内部实现）
    # 仅保留早停回调
    early_stop_callback = EarlyStopping(
        monitor="val/ssim",
        patience=main_cfg['early_stop'],
        mode="max",
        verbose=True
    )
    
    # +++ 新增: ModelCheckpoint回调 +++
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,  # 使用从配置解析的检查点目录
        filename='best_module',
        save_top_k=1,
        monitor="val/ssim",
        mode="max",
        save_last=True,
        every_n_epochs=1
    )

    # 创建导出最佳模型的回调
    # export_model_callback = ExportBestModelCallback(
    #     export_dir=ckpt_dir,
    #     monitor='val/psnr',
    #     mode='max',
    #     start_after_epoch=1,
    # )
    export_metric_callback = ExportMetricCallback(
        export_dir=log_dir,
        monitor='all',
        best_metric_name='val/ssim',
        best_mode='max',
        start_after_epoch=0,
    )

    # 初始化训练器
    trainer = pl.Trainer(
        max_epochs=main_cfg.get('train_epoch', 50),
        accelerator="gpu",
        devices=main_cfg['devices'],
        precision="32-true",  
        # precision="16-mixed",  
        # precision="bf16-mixed", 
        # amp_backend="native",
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=8,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback, checkpoint_callback, LearningRateMonitor(), export_metric_callback],  # 添加ModelCheckpoint
        deterministic=True,
        log_every_n_steps=100,
        enable_progress_bar=True,
        enable_model_summary=True,
        strategy='ddp_find_unused_parameters_true',
        profiler="simple",
        # limit_train_batches=1,
        logger=logger
    )

    torch.use_deterministic_algorithms(mode=True,warn_only=True)
    # 开始训练
    trainer.fit(litmodel, data_module)
    
    # 测试阶段设置测试集
    data_module.setup(stage='test')
    
    #测试最佳模型
    print(f"\n测试最佳模型...{checkpoint_callback.best_model_path}")
    best_model = VGGT4DGS_LITModelModule.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(best_model, data_module)


if __name__ == "__main__":
    main()