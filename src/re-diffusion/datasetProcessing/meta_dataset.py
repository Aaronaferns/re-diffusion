# meta_dataset.py
import json
from torch.utils.data import IterableDataset

class MetaShardDataset(IterableDataset):
    def __init__(self, shard_files, rank, world_size):
        self.shards = shard_files[rank::world_size]

    def __iter__(self):
        for shard in self.shards:
            with open(shard, "r") as f:
                for line in f:
                    yield json.loads(line)
