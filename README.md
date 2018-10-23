# Azure Machine Learning service End-To-End Walkthrough

The goal of the lab is to introduce key components and features of Azure Machine Learning service. During the lab you will go through the full Machine Learning workflow from data preparation, through model training, to model operationalization.

The lab can be delivered as a half-a-day workshop. Follow the [Delivery guide](delivery_guide.md) for the recommended flow of the workshop.

## Lab environment set up


You can use your workstation or Azure Notebooks. We would recommend using you workstation.

### To set up your workstation

1. Follow instructions below to install Anaconda for Python 3

https://www.anaconda.com/

2. Create a new conda environment

```
conda create -n <your env name> Python=3.6 anaconda

# On Linux or MacOs
source activate <your env name>

# On Windows 
activate <your env name>
```

3. Install Azure ML Python SDK
```
pip install --upgrade azureml-sdk[notebooks,automl]
```

4. Configure AML widgets for Jupyter
```
jupyter nbextension install --py --user azureml.train.widgets
jupyter nbextension enable --py --user azureml.train.widgets
```

5. Clone this repo
```
git clone <repo URL>
```

6. Start Jupyter and enjoy
```
jupyter notebook
```


