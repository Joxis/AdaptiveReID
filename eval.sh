python -u eval.py \
  --dataset_name "custom" \
  --root_folder_path "../../Datasets/re-identification" \
  --backbone_model_name "ResNet50" \
  --pretrained_model_file_path "models/Market1501_ResNet50_9502037.h5" \
  --output_folder_path "evaluation_only" \
  --evaluation_only \
  --freeze_backbone_for_N_epochs 0 \
  --testing_size 1.0 \
  --evaluate_testing_every_N_epochs 1