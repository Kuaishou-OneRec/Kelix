from . import BaseStrategy


class ViTStrategy(BaseStrategy):

    def __init__(self, config, ctx, **kwargs):
        super().__init__(config, ctx, **kwargs)

        self.rank = ctx.get('rank', 0)
        self.world_size = ctx.get('world_size', 1)
        self.local_rank = ctx.args.local_rank

    def step(self):
        pass

    def setup(self):
        self.set_random_seed()
