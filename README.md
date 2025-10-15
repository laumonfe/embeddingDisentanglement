# State Space model for multimodal learning using Feidegger

## What is it? 






## The Dataset

The FEIDEGGER dataset consists of 8732 high-resolution images of dresses, each with 5 textual annotations in German. For more information on the dataset, please check out the [dataset's repository](https://github.com/laumonfe/feidegger/tree/master) and the [LREC 2018 paper](http://aclweb.org/anthology/L18-1070). 


## Requirements

Install dependencies:
```
python install_dependencies.py
```
Or with a Conda environment:

```
python install_dependencies.py --conda --env-name feidegger-mamba
```

## Download the dataset 
```
python feidegger_mamba_prep.py --data_path data/FEIDEGGER_release_1.2.json --output_dir mamba_dataset
```

## Citing FEIDEGGER

Please cite the following paper when using FEIDEGGER: 

```
@inproceedings{lefakis2018feidegger,
  title={FEIDEGGER: A Multi-modal Corpus of Fashion Images and Descriptions in German},
  author={Lefakis, Leonidas and Akbik, Alan and Vollgraf, Roland},
  booktitle = {{LREC} 2018, 11th Language Resources and Evaluation Conference},
  year      = {2018}
}
```
