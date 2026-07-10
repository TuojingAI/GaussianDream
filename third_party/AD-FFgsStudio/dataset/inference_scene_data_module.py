import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import partial
import torch

from dataset.data_util import train_transforms, inference_transforms
from dataset.vggt3dgs_scene_dataset_12hz import NuScenesSceneDataset

# 测试v2.0 用下面这个
# from dataset.vggt4dgs_scene_dataset import NuScenesdataset4D as NuScenesSceneDataset


class SceneDataLoader:
    """Custom DataLoader that provides scene-by-scene iteration"""
    
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_scenes = dataset.get_num_scenes()
        
        # Create scene indices for iteration
        self.scene_indices = list(range(self.num_scenes))
        if shuffle:
            import random
            random.shuffle(self.scene_indices)
    
    def __iter__(self):
        """Iterate over scenes, yielding scene metadata only (lazy loading)"""
        for scene_idx in self.scene_indices:
            scene_name = self.dataset.get_scene_name(scene_idx)
            scene_token = self.dataset.get_scene_token(scene_idx)
            scene_length = self.dataset.get_scene_length(scene_idx)
            
            # Return only scene metadata and indices for lazy loading
            # External code can call dataset.get_scene_sample(scene_idx, sample_idx) as needed
            yield {
                'scene_idx': scene_idx,
                'scene_name': scene_name,
                'scene_token': scene_token,
                'scene_length': scene_length,
                'sample_indices': list(range(scene_length)),  # Just indices, not actual data
                'dataset': self.dataset  # Reference to dataset for lazy loading
            }
    
    def __len__(self):
        return self.num_scenes


class VGGT3DGS_SceneDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule with scene-based organization
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.read_config(cfg)

    def read_config(self, cfg):    
        for k, v in cfg.items():
            setattr(self, k, v)

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.construct_dataset('train')
            self.val_dataset = self.construct_dataset('val')

        if stage == "test" or stage is None:
            self.test_dataset = self.construct_dataset('test')

    def train_scene_dataloader(self):
        """Return scene-based dataloader for training"""
        if self.train_dataset is None:
            return None
        return SceneDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.data_shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_scene_dataloader(self):
        """Return scene-based dataloader for validation"""
        if self.val_dataset is None:
            return None
        return SceneDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_scene_dataloader(self):
        """Return scene-based dataloader for testing"""
        if self.test_dataset is None:
            return None
        return SceneDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # Standard DataLoaders for compatibility
    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.data_shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=True,
        )
            
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def construct_dataset(self, mode):
        """Construct scene-based dataset"""
        
        if hasattr(self,'crop_scale'):
            crop_scale = self.crop_scale
        else:
            crop_scale = []
        if hasattr(self,'crop_ratio'):
            crop_ratio = self.crop_ratio
        else:
            crop_ratio = []
        if hasattr(self,'crop_prob'):
            crop_prob = self.crop_prob
        else:
            crop_prob = 0.0
        if hasattr(self,'jittering'):
            jittering = self.jittering
        else:
            jittering = []
        if hasattr(self,'jittering_prob'):
            jittering_prob = self.jittering_prob
        else:
            jittering_prob = 0.0

        # Use different transforms for train vs inference
        if mode == 'train':
            data_transform = partial(train_transforms,
                       image_shape=(int(self.height), int(self.width)),
                       crop_scale=crop_scale,
                       crop_ratio=crop_ratio,
                       crop_prob=crop_prob,
                       jittering=jittering,
                       jittering_prob=jittering_prob)
        else:
            # Use optimized inference transforms for val/test
            data_transform = partial(inference_transforms,
                       image_shape=(int(self.height), int(self.width)))
        
        dataset_args = {
            'cameras': self.cameras,
            'back_context': self.back_context,
            'forward_context': self.forward_context,
            'data_transform': data_transform,
            'depth_type': self.depth_type if 'gt_depth' in self.train_requirements else None,
            'with_pose': 'gt_pose' in self.train_requirements,
            'with_ego_pose': 'gt_ego_pose' in self.train_requirements,
            'with_mask': 'mask' in self.train_requirements,
            'test_scenes': getattr(self, 'test_scenes', None)
        }
        
        dataset = NuScenesSceneDataset(self.data_path, mode, **dataset_args)

        return dataset


# Helper functions for external scene-based training
def train_scene_based_epoch(model, scene_dataloader, optimizer, device='cuda'):
    """
    Train one epoch with scene-based iteration
    
    Args:
        model: The model to train
        scene_dataloader: SceneDataLoader instance
        optimizer: Optimizer
        device: Device to use
    
    Returns:
        List of scene results
    """
    model.train()
    scene_results = []
    
    for scene_batch in scene_dataloader:
        scene_idx = scene_batch['scene_idx']
        scene_name = scene_batch['scene_name']
        scene_samples = scene_batch['samples']
        
        print(f"Training on scene {scene_idx}: {scene_name} ({len(scene_samples)} samples)")
        
        # Process all samples in this scene
        scene_losses = []
        for sample_idx, sample in enumerate(scene_samples):
            # Move sample to device
            for key, value in sample.items():
                if torch.is_tensor(value):
                    sample[key] = value.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(sample)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            scene_losses.append(loss.item())
        
        scene_result = {
            'scene_idx': scene_idx,
            'scene_name': scene_name,
            'avg_loss': sum(scene_losses) / len(scene_losses),
            'num_samples': len(scene_samples)
        }
        scene_results.append(scene_result)
        
        print(f"Scene {scene_name} - Avg Loss: {scene_result['avg_loss']:.4f}")
    
    return scene_results


def evaluate_scene_based(model, scene_dataloader, device='cuda'):
    """
    Evaluate model with scene-based iteration
    
    Args:
        model: The model to evaluate
        scene_dataloader: SceneDataLoader instance
        device: Device to use
    
    Returns:
        List of scene results
    """
    model.eval()
    scene_results = []
    
    with torch.no_grad():
        for scene_batch in scene_dataloader:
            scene_idx = scene_batch['scene_idx']
            scene_name = scene_batch['scene_name']
            scene_samples = scene_batch['samples']
            
            print(f"Evaluating scene {scene_idx}: {scene_name} ({len(scene_samples)} samples)")
            
            # Process all samples in this scene
            scene_outputs = []
            for sample_idx, sample in enumerate(scene_samples):
                # Move sample to device
                for key, value in sample.items():
                    if torch.is_tensor(value):
                        sample[key] = value.to(device)
                
                # Forward pass
                outputs = model(sample)
                scene_outputs.append(outputs)
            
            scene_result = {
                'scene_idx': scene_idx,
                'scene_name': scene_name,
                'scene_token': scene_batch['scene_token'],
                'outputs': scene_outputs,
                'num_samples': len(scene_samples)
            }
            scene_results.append(scene_result)
            
            print(f"Scene {scene_name} - {len(scene_outputs)} outputs collected")
    
    return scene_results


if __name__ == '__main__':
    import yaml
    
    # Test the scene-based data module
    config_file = 'configs/nuscenes/vggt3dgs.yaml'
    with open(config_file) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    scene_datamodule = VGGT3DGS_SceneDataModule(main_cfg['data_cfg'])
    scene_datamodule.setup('test')
    scene_dataset = scene_datamodule.test_dataset
    # with open('eval_scenes.txt', 'w') as fw:
    #     for scene_name, scene_data in zip(scene_dataset.scene_names,scene_dataset.scenes_data):
    #         print(scene_name,scene_data)
    #         for sample_token in scene_data:
    #             fw.write(sample_token+'\n')
    # Test scene-based dataloader
    scene_loader = scene_datamodule.val_scene_dataloader()
    
    print(f"Number of scenes in validation set: {len(scene_loader)}")
    
    # Test iteration over first few scenes
    for i, scene_batch in enumerate(scene_loader):
        if i >= 2:  # Only test first 2 scenes
            break
        
        print(f"\nScene {i}:")
        print(f"  Name: {scene_batch['scene_name']}")
        print(f"  Token: {scene_batch['scene_token']}")
        print(f"  Length: {scene_batch['scene_length']}")
        print(f"  Number of samples loaded: {len(scene_batch['samples'])}")
        
        # Check first sample structure
        if scene_batch['samples']:
            first_sample = scene_batch['samples'][0]
            print(f"  First sample keys: {list(first_sample.keys())}")