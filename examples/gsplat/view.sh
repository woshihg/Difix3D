#SCENE_ID=5c3af581028068a3c402c7cbe16ecf9471ddf2897c34ab634b7b1b6cf81aba00
#DATA=/home/woshihg/PycharmProjects/Difix3D/DL3DV-10K/${SCENE_ID}/colmap
#CKPT_PATH=/home/woshihg/PycharmProjects/Difix3D/outputs/difix3d/gsplat/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt
#DATA_FACTOR=4
#OUTPUT_DIR=outputs/difix3d/gsplat/${SCENE_ID}
#CUDA_VISIBLE_DEVICES=0 python examples/gsplat/visualization.py default \
#    --data_dir ${DATA} --data_factor ${DATA_FACTOR} \
#    --ckpt ${CKPT_PATH} \
#    --result_dir ${OUTPUT_DIR} --no-normalize-world-space --test_every 4

SCENE_ID=TEST
DATA=/home/woshihg/PycharmProjects/Difix3D/dataset/${SCENE_ID}
CKPT_PATH=/home/woshihg/PycharmProjects/Difix3D/outputs/difix3d/gsplat/densepoints/300/${SCENE_ID}/ckpts/ckpt_9999_rank0.pt
DATA_FACTOR=1
TEST_EVERY=100
OUTPUT_DIR=outputs/difix3d/gsplat/densepoints/${TEST_EVERY}/${SCENE_ID}
CUDA_VISIBLE_DEVICES=0 python examples/gsplat/visualization.py default \
    --data_dir ${DATA} --data_factor ${DATA_FACTOR} \
    --ckpt ${CKPT_PATH} \
    --result_dir ${OUTPUT_DIR} --no-normalize-world-space --test_every ${TEST_EVERY}