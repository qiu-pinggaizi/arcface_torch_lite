# Navigate to project root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_v2.py configs/glint360k_mbf_v2.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_v2.py configs/glint360k_r18.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_v2.py configs/glint360k_mbf_large.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_v2.py configs/glint360k_mbf_v2.py
