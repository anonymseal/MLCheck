
# MLCheck

This repository contains the necessary codes and datasets to replicate the results mentioned in our paper. You need to install conda to run our tool. Here is a [link](https://docs.anaconda.com/anaconda/install/) to know how to install anaconda. After installing anaconda into your system you need create a virtual conda environment. You can do so by running the following command:
```
conda create -n myenv python=3.6
```
When it asks you to proceed 
```
proceed ([y]/n)?
```
Then type y. After that you can activate the environment by running:
```
conda activate myenv
```
After activating the new conda environment, you will need to install the required packages to run our tool. You can run the following command to install the necessary packages:
```
pip install requirements.txt
```
After installing the necessary packages, you can run several commands to see our results. As our tool need a significant amount of time to run all the test cases, we don't show the results for all the test cases mentioned in the paper. To replicate the results of Table 4 for the test cases Decision tree, Logistic Regression, Naive Bayes, Fair-Aware1 and Fair-Aware2 for both the Adult and Credit dataset, please run the following command
```
python test_function_fairness.py 

```
If you would like to replicate the results of Table 6 for the test cases CR1-CR4, please run the following command:
```
python test_function_embeddings.py 

```
If you would like to replicate the results of Table 7 from T1-4 to T3-5, please run the following command:

```
python test_function_trojan_1K.py 

```
If you would like to replicate the results of Table 78from T1-4 to T3-5, please run the following command:

```
python test_function_trojan_10K.py 

```

Please keep in mind these cases take a significant amount of time to produce the results. All the results are written in a ```.txt``` file inside the ```Òutput\``` folder
