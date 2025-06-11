import argparse

from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig

from mcore_adapter.models.converter.post_converter import convert_checkpoint_to_hf  
import torch

parser = argparse.ArgumentParser(description="PPO Configuration")

parser.add_argument(
    "--config_name", type=str, default="rlvr_megatron_config", help="Name of the PPO configuration."
)

parser.add_argument(
    "--checkpoint_path", type=str, default=None, help="Path for which the checkpoint is saved."
)

parser.add_argument(
    "--output_huggingface_pretrain_path", type=str, default=None, help="Path for which the converted checkpoint is to be saved."
)

args = parser.parse_args()


def make_ppo_config():

    config_path = "."
    config_name = args.config_name

    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    return ppo_config


def test_make_ppo_config():
    ppo_config = make_ppo_config()
    print(ppo_config)


def test_ppo_pipeline():

    ppo_config = make_ppo_config()

    init()

    from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline
    pipeline = RLVRPipeline(pipeline_config=ppo_config)

    pipeline.run()

def convert_checkpoint(checkpoint_path, output_path):  
    torch_dtype = torch.float16  # or torch.bfloat16  
      
    convert_checkpoint_to_hf(  
        model_name_or_path=checkpoint_path,  
        save_directory=output_path,  
        torch_dtype=torch_dtype,  
        verbose=True  
    )



if __name__ == "__main__":
    if args.checkpoint_path and args.output_huggingface_pretrain_path:
        convert_checkpoint(args.checkpoint_path, args.output_huggingface_pretrain_path)
    test_ppo_pipeline()
