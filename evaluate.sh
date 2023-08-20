# C+T -> S/K
# For EMD-S
python evaluate.py --model weights/EMD_things_S.pth --dataset sintel --iters8 18 --iters4 0 --model_type S
python evaluate.py --model weights/EMD_things_S.pth --dataset kitti --iters8 18 --iters4 0 --model_type S

# For EMD-M
python evaluate.py --model weights/EMD_things_M.pth --dataset sintel --iters8 18 --iters4 6 --model_type M
python evaluate.py --model weights/EMD_things_M.pth --dataset kitti --iters8 18 --iters4 6 --model_type M

# For EMD-M
python evaluate.py --model weights/EMD_things_L.pth --dataset sintel --iters8 18 --iters4 6 --model_type L
python evaluate.py --model weights/EMD_things_L.pth --dataset kitti --iters8 18 --iters4 6 --model_type L
