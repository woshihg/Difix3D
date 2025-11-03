#SCENE_ID=5c3af581028068a3c402c7cbe16ecf9471ddf2897c34ab634b7b1b6cf81aba00
#DATA=/home/woshihg/PycharmProjects/Difix3D/DL3DV-10K/${SCENE_ID}/colmap
#CKPT_PATH=/home/woshihg/PycharmProjects/Difix3D/outputs/difix3d/gsplat/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt
#DATA_FACTOR=4
#OUTPUT_DIR=outputs/difix3d/gsplat/100/${SCENE_ID}
#CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
#    --data_dir ${DATA} --data_factor ${DATA_FACTOR} --disable_viewer\
#    --result_dir ${OUTPUT_DIR} --no-normalize-world-space --test_every 100
##        --ckpt ${CKPT_PATH} \

SCENE_ID=TEST
DATA=/home/woshihg/PycharmProjects/Difix3D/dataset/${SCENE_ID}
CKPT_PATH=/home/woshihg/PycharmProjects/Difix3D/outputs/difix3d/gsplat/${SCENE_ID}/ckpts/ckpt_29999_rank0.pt
DATA_FACTOR=1
TEST_EVERY=51
TRAIN_SEQUNCES=[1,16,135,255]
OUTPUT_DIR=outputs/difix3d/gsplat/pointsgroup/${TEST_EVERY}/${SCENE_ID}
CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA} --data_factor ${DATA_FACTOR} \
    --result_dir ${OUTPUT_DIR} --no-normalize-world-space --test_every ${TEST_EVERY} \
    -- train_sequences ${TRAIN_SEQUNCES} \
    #        --ckpt ${CKPT_PATH} \
#    --disable_viewer\
