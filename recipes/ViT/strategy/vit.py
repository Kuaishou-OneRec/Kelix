from . import BaseStrategy


class ViTStrategy(BaseStrategy):

    def __init__(self, config, ctx, **kwargs):
        super().__init__(config, ctx, **kwargs)

        self.rank = kwargs.get('rank', 0)
        self.world_size = kwargs.get('world_size', 1)

    def step(self):
        pass

    def setup(self):
        pass
