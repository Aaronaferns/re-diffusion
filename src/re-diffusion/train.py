import glob
import torchvision.transforms as T
from ddp import setup_ddp
from datasetProcessing.meta_dataset import MetaShardDataset
from datasetProcessing.image_cache import ImageCache
from datasetProcessing.laion_stream import LAIONStream
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

rank, world_size = setup_ddp()

# ---- METADATA SHARDS (NETWORK STORAGE OK) ----
meta_files = sorted(glob.glob("meta_shards/meta-*.jsonl"))

meta_ds = MetaShardDataset(
    shard_files=meta_files,
    rank=rank,
    world_size=world_size,
)

# ---- LOCAL CACHE (CRITICAL) ----
cache = ImageCache(
    cache_dir=f"/tmp/laion_cache_{rank}",  # RAM-backed on dl.luddy
    timeout=5
)

dataset = LAIONStream(meta_ds, cache)

loader = create_loader(dataset, batch_size=8)

# ---- MODEL ----
model = MyModel().cuda()
model = DDP(model, device_ids=[rank])

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
])

# ---- TRAIN LOOP (NO EPOCHS) ----
for step, (images, captions) in enumerate(loader):
    images = torch.stack([transform(img) for img in images])
    images = images.cuda(non_blocking=True)

    loss = model(images, captions)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 100 == 0 and rank == 0:
        print(f"step {step} loss {loss.item()}")
