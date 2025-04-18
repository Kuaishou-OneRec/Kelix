import os
import torch
import torch.nn as nn
import os.path as osp
import torch.distributed as dist
from . import BaseStrategy
import logging
logger = logging.getLogger(__name__)


class ViTStrategy(BaseStrategy):

    def __init__(self, config, ctx, **kwargs):
        super().__init__(config, ctx, **kwargs)

        self.rank = ctx.get('rank', 0)
        self.world_size = ctx.get('world_size', 1)
        self.local_rank = ctx.args.local_rank

        self.save_per_step = config.strategy.save_per_step

    def step(self, ctx):
        self.save(ctx)

    def setup(self):
        self.set_random_seed()
        self.resume()

    def resume(self):
        pass

    def save(self, ctx, output_dir=None):
        if ctx.step % self.save_per_step != 0:
            return
        output_dir = output_dir or self.config.output_dir
        model = ctx.model
        dataloader = ctx.dataloader
    
        model.save_checkpoint(
            save_dir=output_dir,
            client_state={
                "total_num_tokens": ctx.total_num_tokens,
                "total_num_samples": ctx.total_num_samples,

                "total_text_num_tokens": ctx.total_text_num_tokens,
                "total_text_num_valid_tokens": ctx.total_text_num_valid_tokens,

                "total_image_num_tokens": ctx.total_image_num_tokens
            }
        )
        try:
            dataloader_state_dict = {
                "dataloader_state_dict": dataloader.state_dict()
            }
        except:
            dataloader_state_dict = None
            logging.error(f"Dataloader cannot dump state_dict!!!!!!!!")
        
        if dataloader_state_dict is not None:
            dataloader_path = osp.join(output_dir, "dataloader_ckpt")
            if self.rank == 0:
                os.makedirs(dataloader_path, exist_ok=True)
            dist.barrier()
            torch.save(
                dataloader_state_dict,
                osp.join(
                    dataloader_path,
                    f"rank{self.rank}_global_step{ctx.step}.pth"
                )
            )
