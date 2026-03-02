from huggingface_hub import snapshot_download

HF_HOME = "./Anole-zebra-cot"
repo_id = "multimodal-reasoning-lab/Anole-Zebra-CoT"

snapshot_download(
    repo_id=repo_id,
    local_dir=HF_HOME,  # 明确指定下载目录
    local_dir_use_symlinks=False,
    resume_download=True
)