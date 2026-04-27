# Navigate to project root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29522 --nproc_per_node=4 train_v2_prune.py \
        --config configs/glint360k_mbf_v3.py
