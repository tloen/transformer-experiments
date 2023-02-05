from implementations.base import GPTConfig
from implementations.memoized import MemoizedGPT


class FP16MemoizedGPT(MemoizedGPT):
    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.half()
        
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        gpt = super().from_pretrained(*args, **kwargs)
        gpt.half()
        return gpt