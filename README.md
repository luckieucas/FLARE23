## Environments and Requirements

- Ubuntu 20.04 LTS
- RTX 3090 with 24GB GPU memory
- 11.7
- Python 3.10

To install requirements:

```setup
git clone 
cd nnUNet
pip install -e .
```

>Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...



## Dataset

- MICCAI FLARE 2023 dataset

## Preprocessing

A brief description of the preprocessing method

- cropping
- intensity normalization
- resampling

Running the data preprocessing code:

```python
python preprocessing.py --input_path <path_to_input_data> --output_path <path_to_output_data>
```

## Training

1. To train the model(s) in the paper, run this command:

```bash
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>Describe how to train the models, with example commands, including the full training procedure and appropriate hyper-parameters.

You can download trained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on the above dataset with the above code. 

>Give a link to where/how the trained models can be downloaded.


2. To fine-tune the model on a customized dataset, run this command:

```bash
python finetune.py --input-data <path_to_data> --pre_trained_model_path <path to pre-trained model> --other_flags
```

3. [Colab](https://colab.research.google.com/) jupyter notebook


## Inference

1. To infer the testing cases, run this command:

```python
python inference.py --input-data <path_to_data> --model_path <path_to_trained_model> --output_path <path_to_output_data>
```

> Describe how to infer testing cases with the trained models.

2. [Colab](https://colab.research.google.com/) jupyter notebook

3. Docker containers on [DockerHub](https://hub.docker.com/)

```bash
docker container run --gpus "device=0" -m 28G --name algorithm --rm -v $PWD/CellSeg_Test/:/workspace/inputs/ -v $PWD/algorithm_results/:/workspace/outputs/ algorithm:latest /bin/bash -c "sh predict.sh"
```

## Evaluation

To compute the evaluation metrics, run:

```eval
python eval.py --seg_data <path_to_inference_results> --gt_data <path_to_ground_truth>
```

>Describe how to evaluate the inference results and obtain the reported results in the paper.



## Results

Our method achieves the following performance on [Brain Tumor Segmentation (BraTS) Challenge](https://www.med.upenn.edu/cbica/brats2020/)

| Model name       |  DICE  | 95% Hausdorff Distance |
| ---------------- | :----: | :--------------------: |
| My awesome model | 90.68% |         32.71          |

>Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>Pick a license and describe how to contribute to your code repository. 

## Acknowledgement

> We thank the contributors of public datasets. 


# 3. How to ensure reproducibility when the data cannot be shared?

- Try to find a related public dataset to evaluate your method, e.g., [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/), [grand-challenge](https://grand-challenge.org/challenges/). If none of the public datasets can be used to evaluate your method, please explicitly claim it, which is very helpful for communities to create task-driven public datasets.
- Create a docker container to pack and share the proposed method. Many MICCAI challenges have used the docker as the submission, e.g., [ADAM](http://adam.isi.uu.nl/methods/submit/), [M&Ms](https://www.ub.edu/mnms/)
