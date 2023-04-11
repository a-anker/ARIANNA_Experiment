# Deep Learning in ARIANNA

## Setting up the environment
The steps listed below are how to get a similar environment to mine when conducting this research. There are however many ways to set up similar environments so use this as only one reference.

- Download and install [Anaconda](https://www.anaconda.com/products/distribution)
- Create conda environment with a user "chosen_name" and then list all environments
            
      $ conda create --name chosen_name
      $ conda env list
            
- Activate Environment

      $ conda activate chosen_name
                   
- Deactivate Environment

      $ conda deactivate
      
When the conda environment is activated, packages will be installed only in this environment. Activate the environment once again then install the pip package manager and other relevant python packages. I use tensorflow 2 and python 3.8.5.
      
      $ conda install pip
      $ pip install matplotlib numpy scipy tensorflow keras, parameter-sherpa 
            
Try importing some of these packages to determine if the packages were downloaded sucessfully. If there are any dependencies missing from this list, just pip install them the same way as above.

      $ python
      >>> from matplotlib import pyplot as plt
      >>> import numpy as np
      >>> import scipy 
      >>> import keras
      >>> import tensorflow
      
Next to download this repository to a local computer, use cd to go into your chosen directory for this analysis then
            
      $ git clone https://github.com/a-anker/ARIANNA_Experiment.git
      
      
All of these scripts use the os.environ['ARIANNA_Experiment'] variable to locate this github repository and its scripts, so add your specified local path as an environment variable. To do this, locate and open your .bashrc or .bash_profile, which is usually in your home directory (nano ~/.bash_profile). Next input your github repository location as an environment variable. For example, if this downloaded github repository is located in /home/aanker, you will need to add the text below to the .bash_profile file:

      export ARIANNA_Experiment="/home/aanker/ARIANNA_Experiment"      
    
This will ensure that the script in this repository will be found by your local computer.
      
      
      
      
      
      
For more tips on installing other analysis tools used within ARIANNA such as ROOT, snowShovel, and additional requirements, refer to the README in Geoffrey Gaswint's [ARIANNA](https://github.com/ggaswint/ARIANNAanalysis) repository.
      
      

# Plots and their corresponding scripts


**Refer to template_study/:**

<img src="https://user-images.githubusercontent.com/38436394/226751873-e69ddeda-e275-4d9c-9a43-69f3ae935690.png" width=50% height=50%>


**Refer to interpretability/:**

<p float="left">
  <img src="https://user-images.githubusercontent.com/38436394/226752913-3067bfc4-40c7-4ae7-8930-43de1f14130b.png" width="250" />
  <img src="https://user-images.githubusercontent.com/38436394/226752917-caf32021-a495-4572-889b-6427492dee29.png" width="250" /> 
  <img src="https://user-images.githubusercontent.com/38436394/226752918-b90f2b9a-421a-447e-a6df-6a499d88226b.png" width="250" />
</p>

**Refer to 5-fold_CV/:**

<img src="https://user-images.githubusercontent.com/38436394/226755074-339cc62d-49bd-40db-9c1f-1ee4fe212466.png" width=50% height=50%>

**Refer to cnn_train_test_efficiency.py:**

<img src="https://user-images.githubusercontent.com/38436394/226754574-fc267146-9a36-40de-ade5-fd144383f845.png" width=50% height=50%>

**Refer to train_cnn_with_acc_loss_plot.py:**

<p float="left">
  <img src="https://user-images.githubusercontent.com/38436394/226755185-840b8a01-078c-42bf-9cbb-dfe009572a72.png" width="400" />
  <img src="https://user-images.githubusercontent.com/38436394/226755206-fbed3678-341e-45a4-8a6f-369a63c2dbf8.png" width="400" /> 
</p>

**Refer to correlation/:**

<img src="https://user-images.githubusercontent.com/38436394/226755525-0d19fc7f-17a5-49be-b672-466a176f32d2.png" width=50% height=50%>

**Refer to compare_similar_distributions/:**

<img src="https://user-images.githubusercontent.com/38436394/226755679-fb3a3ead-604f-4a25-9a92-7bf390f2768c.png" width=50% height=50%>

**Refer to fft.py:**

<img src="https://user-images.githubusercontent.com/38436394/226755781-0a914f17-d640-4411-9599-edb7f0713f0a.png" width=50% height=50%>


