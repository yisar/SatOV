import os
from huggingface_hub import snapshot_download

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

snapshot_download(
    repo_id="buckets/yisar/SegEarth-OV",
    local_dir=r"F:\www\RSOVS\SegEarth-OV",
    force_download=False
)