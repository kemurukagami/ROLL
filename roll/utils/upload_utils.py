import os
import shutil

from roll.utils.logging import get_logger


logger = get_logger()


class FileSystemUploader:
    """
    将本地的ckpt目录上传到文件系统, oss/cpfs在多Role的场景下，
    每个Role会把自己的ckpt dir的内容上传到OUTPUT_DIR/ckpt_id/下
    {
        "type": "file_system",
        "output_dir": /data/oss_bucket_0/llm/models
    }
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        logger.info(f"use FileSystemUploader to upload {output_dir}")

    def upload(self, ckpt_id: str, local_state_path: str, **kwargs):
        ckpt_id_output_dir = os.path.join(self.output_dir, ckpt_id)
        os.makedirs(ckpt_id_output_dir, exist_ok=True)
        logger.info(f"{local_state_path} save to {ckpt_id_output_dir}, wait...")
        shutil.copytree(local_state_path, ckpt_id_output_dir, dirs_exist_ok=True)
        logger.info(f"{local_state_path} save to {ckpt_id_output_dir}, done...")
    
    def upload_file(self, ckpt_id: str, local_file_path: str, relative_path: str = None, **kwargs):  
        """Upload a single file to the checkpoint directory"""  
        ckpt_id_output_dir = os.path.join(self.output_dir, ckpt_id)  
        os.makedirs(ckpt_id_output_dir, exist_ok=True)  
          
        if relative_path:  
            target_dir = os.path.join(ckpt_id_output_dir, os.path.dirname(relative_path))  
            os.makedirs(target_dir, exist_ok=True)  
            target_path = os.path.join(ckpt_id_output_dir, relative_path)  
        else:  
            target_path = os.path.join(ckpt_id_output_dir, os.path.basename(local_file_path))  
          
        logger.info(f"Uploading {local_file_path} to {target_path}")  
        shutil.copy2(local_file_path, target_path)  
        logger.info(f"Upload complete: {target_path}")
