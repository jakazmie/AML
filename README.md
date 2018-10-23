# Azure Machine Learning service End-To-End Walkthrough

The goal of the lab is to introduce key components and features of Azure Machine Learning service. During the lab you will go through the full Machine Learning workflow from data preparation, through model training, to model operationalization.

The lab can be delivered as a half-a-day workshop. Follow the [Delivery guide](delivery_guide.md) for the recommended flow of the workshop.

## Lab environment set up


You can use your workstation or Azure Data Science Virtual Machine as your lab environment.

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

5. Install TensorFlow
```
pip install --upgrade tensorflow
# pip install --upgrade tensorflow-gpu
```

6. Clone this repo
```
git clone <repo URL>
```

7. Start Jupyter and enjoy
```
jupyter notebook
```




### To set up Azure Data Science Virtual Machine

1. Follow the below link to provision Data Science Virtual Machine. 
   - Create a new resource group for your VM
   - Use **DS3_v2** or better as a VM type. Although, the labs will run on other configurations this is the minimum configuration we recommend. 
   - Choose *username and password* as the authentication type. 
   - Use default values for all other parameters.

 https://portal.azure.com/#create/microsoft-dsvm.linux-data-science-vm-ubuntulinuxdsvmubuntu

2. When your VM is ready use Azure Portal Cloud Shell to complete the configuration

```
# Logon to your VM
ssh <your username>@<vm ip address>

# Clone the labs in the notebooks folder
cd notebooks
git clone <repo ULR>

exit
```

3. Use Chrome browser to connect to Jupyter Hub at `http://<your machine's IP address>:8000`. You may receive a warning that `Your connection is not private`. Ignore it and press on **ADVANCED** to proceed.

3. Use your username and password to log in to Jupyter

4. The next step is a temporary walk-around to address issues with AML Widgets on Azure Data Science Machine
```
# Log back to your DSVM
ssh <your username>@<vm ip address>

# Enable AML widgets. This is to address the bug in DSVM default configuration
source activate py36
jupyter nbextension enable --py --user azureml.train.widgets
```

**Important**. Make sure to set the kernel of each notebook in the lab to *Python 3.6 - AzureML*.


Enjoy.

