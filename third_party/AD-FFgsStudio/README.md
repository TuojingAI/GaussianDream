# AD-FFgsStudio

# SplatScene
SplatScene: Feed-Forward 3D Gaussian Splatting Generation for Large-Scale Urban Scenes

## Envs

### Create Conda Envs

  conda create -n vggt-scene python 3.10

### Install Packages (Automatically)

  python -m pip install -r envs/envs_requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

### Install Packages (Manually)

  # pytorch3d
  wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2
  conda install ./pytorch3d-0.7.8-py310_cu121_pyt231.tar.bz2

  # gaussian splatting
  git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive
  cd gaussian-splatting
  pip install submodules/diff-gaussian-rasterization
  pip install submodules/simple-knn
  pip install submodules/fused-ssim

### 快速开始

#### 环境
使用以下命令启动python环境：
source envs/envs_infrawaves.sh

#### 代码解释
这个分支的代码不包含v2.0的vggt4dgs，代码中的vggt4dgs都是v2.1的研发代码




#### 训练模型

使用以下命令启动模型训练：
sh train.sh

#### 推理模型
使用以下命令启动模型推理：  
Driving Forward: bash df3dgs_inference.sh
v2.0-VGGT3dGS: bash vggt3dgs_inference.sh


sh文件内需要调整的主要参数是:  
CONFIG_PATH 表示模型的cfg文件路径
CHECKPOINT_PATH 表示待测评的模型文件
OUTPUT_DIR 模型的可视化保存路径
DEVICE='0'  可以用List传入多卡

#### 评估模型
由于各个版本之间保存的数据帧数不同，统一对齐到vggt4dgs的228帧，在这之后的帧数不进行测评
test.py 基于OUTPUT_DIR下的所有图像进行定量相似度测评和下游感知测评





