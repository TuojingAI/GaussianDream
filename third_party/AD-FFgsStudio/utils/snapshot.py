import os
import shutil
import fnmatch

def save_pipeline_snapshot(snapshot_dir=""):
    """
    保存完整的pipeline快照(代码+模型)
    
    :param model_dir: 快照存储目录
    """
    # 创建版本化目录
    if os.path.exists(snapshot_dir):
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # 定义需要包含的代码文件类型
    INCLUDE_PATTERNS = ['*.py', '*.yaml', '*.sh', '*.txt', '*_mask.png']
    
    project_root = os.getcwd()
    # 自动扫描项目目录
    for root, dirs, files in os.walk(project_root):
        # 排除隐藏文件夹
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            # 检查文件是否匹配需要包含的模式
            if any(fnmatch.fnmatch(file, pattern) for pattern in INCLUDE_PATTERNS):
                file_path = os.path.join(root, file)
                try:
                    # 保持原始目录结构
                    rel_path = os.path.relpath(file_path, start=project_root)
                    dest_path = os.path.join(snapshot_dir, rel_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(file_path, dest_path)
                except Exception as e:
                    print(f"Error copying {file_path}: {str(e)}")
    
    return snapshot_dir




if __name__ == "__main__":
    # 使用示例

    save_pipeline_snapshot(snapshot_dir=os.environ.get('AD_FFGS_SNAPSHOT_DIR', 'snapshot/code'))