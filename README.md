# Learning to Generate Faithful Argumentative Explanations for Fact Check Predictions

The research project has two parts: 

1. To train a model to generate argumentative explanations for the automated fact checking task (this amounts to structured prediction). 
2. To optimise the generation of explanations with respect to the dialectical faithfulness property (this is explained in the first paper listed below).


## Getting Started

```commandline
conda create -n evaluate_explanations python=3.10
conda activate evaluate_explanations
pip install -r requirements.txt
```

## Data

The datasets we will be exploring in this project can be found in `data/`. 

We will be using the VitaminC data for this work. You can discover more about this dataset [here](https://github.com/TalSchuster/VitaminC?tab=readme-ov-file).


## Using metrics

In order to use the metrics defined in this repository add the following line to your python script
```python 
   from src.metrics import argumentative_metrics, deductive_metrics, freeform_metrics
```

See examples in `src/examples`.



 

 