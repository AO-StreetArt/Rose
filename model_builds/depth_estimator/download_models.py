from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Intel/dpt-large",
    local_dir="/app/models/dpt-large",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="Intel/zoedepth-nyu-kitti",
    local_dir="/app/models/zoedepth-nyu-kitti",
    local_dir_use_symlinks=False,
)