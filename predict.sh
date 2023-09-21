export nnUNet_results="$PWD/nnUNet_results/"
export nnUNet_raw="$PWD/nnUNet_raw/"
export nnUNet_preprocessed="$PWD/nnUNet_preprocessed/"

# nnUNetv2_predict -i /workspace/inputs/ -o /workspace/outputs -d 2 -c 3d_midres \
#  -f 1 -chk checkpoint_best.pth --continue_prediction \
#  -tr nnUNetTrainerFlareMergeProb -step_size 0.8 -npp 1 -nps 1


# nnUNetv2_predict -i /workspace/inputs/ -o /workspace/outputs -d 2 -c 3d_verylowres \
#  -f 1 -chk checkpoint_final.pth \
#  -tr nnUNetTrainerFlareMergeProb  -step_size 0.8 -npp 2 -nps 3 -p nnUNetPlansSmall --disable_tta


nnUNetv2_predict -i /workspace/inputs/ -o /workspace/outputs  -d 12 -c 3d_mylowres \
-f 1 -chk checkpoint_best_ep1165.pth \
-tr  nnUNetTrainerFlarePseudoCutUnsupLow -step_size 0.6 -npp 3 --disable_tta