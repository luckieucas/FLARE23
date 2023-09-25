## Environments and Requirements

- Ubuntu 20.04 LTS
- RTX 3090 with 24GB GPU memory
- 11.7
- Python 3.10

To install requirements:

```setup
git clone https://github.com/luckieucas/FLARE23.git
cd nnUNet
pip install -e .
```

>Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...



## Dataset

- MICCAI FLARE 2023 dataset

## Preprocessing


- cropping
- intensity normalization
- resampling
- flip
- rotation
- scale

Running the data preprocessing code:

```bash
nnUNetv2_plan_and_preprocess -d 12 --verify_dataset_integrity
```

## Training

1. To train the model(s) in the paper, run this command:

```bash
python run_training_Flare.py 12 3d_mylowres 1 -tr nnUNetTrainerFlarePseudoCutUnsupLow -p nnUNetPlans
```


## Inference

1. To infer the testing cases, run this command:

```python
nnUNetv2_predict -i <path_to_data> -o  <path_to_output_data>  -d 12 -c 3d_mylowres -f 1 -chk <name_of_trained_model> -tr  nnUNetTrainerFlarePseudoCutUnsupLow -step_size 0.6 -npp 3 --disable_tta
```

## Evaluation

To compute the evaluation metrics, run:

```eval
python eval.py --seg_data <path_to_inference_results> --gt_data <path_to_ground_truth>
```

>Describe how to evaluate the inference results and obtain the reported results in the paper.



## Results

Our method achieves the following performance on [MICCAI FLARE23: Fast, Low-resource, and Accurate oRgan and Pan-cancer sEgmentation in Abdomen CT](https://codalab.lisn.upsaclay.fr/competitions/12239)

| Model name       |  DICE  | 95% Hausdorff Distance |
| ---------------- | :----: | :--------------------: |
| My awesome model | 90.68% |         32.71          |

>Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>Pick a license and describe how to contribute to your code repository. 

## Acknowledgement

> We thank the contributors of public datasets. 
