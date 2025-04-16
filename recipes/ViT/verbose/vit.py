from . import BaseVerbose


class ViTVerbose(BaseVerbose):

    _default_print_prompt = "RANK[{}]"

    def __init__(self, config, ctx, **kwargs):
        super().__init__(config, ctx, **kwargs)

        self.rank = ctx.get('rank', 0)
        self.world_size = ctx.get('world_size', 1)
        self.local_rank = ctx.args.local_rank

    def setup(self):
        pass

    def register_hooks(self):
        pass

    def print(self, *args, **kwargs):
        rank = kwargs.pop('rank', 0)
        prompt = kwargs.pop('prompt', self._default_print_prompt)

        if self.rank == rank or rank == -1:
            if prompt == "":
                print(*args, **kwargs)
            elif prompt == self._default_print_prompt:
                print(prompt.format(self.rank), *args, **kwargs)
            else:
                print(prompt, *args, **kwargs)

    def step(self):
        pass
