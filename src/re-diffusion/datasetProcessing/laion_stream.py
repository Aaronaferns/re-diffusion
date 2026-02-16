# laion_stream.py
from torch.utils.data import IterableDataset

class LAIONStream(IterableDataset):
    def __init__(self, meta_dataset, cache):
        self.meta_dataset = meta_dataset
        self.cache = cache

    def __iter__(self):
        for meta in self.meta_dataset:
            img = self.cache.get(meta["url"])
            if img is None:
                continue
            yield img, meta["caption"]
