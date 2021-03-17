# English-to-Python-Converter
This is an attempt to use transformers and self attention in order to convert English descriptions into Python code.

[Notebook](https://github.com/divyam96/English-to-Python-Converter/blob/main/English_to_Python.ipynb) 

[Pretrained Model](https://drive.google.com/file/d/1-YLd_DTt3W8R_vqga70zdJ8pdK3x2nWm/view?usp=sharing) 

[Dataset](https://drive.google.com/file/d/1rHb0FQ5z5ZpaY2HpyFGY6CeyDG0kTLoO/view?usp=sharing)

## Data Cleaning

We will be using this precurated [Dataset](https://drive.google.com/file/d/1rHb0FQ5z5ZpaY2HpyFGY6CeyDG0kTLoO/view?usp=sharing) for training our transformer model. The format of the data is as follows:

```
# English Description 1

<Python Code 1>

# English Description 2

<Python Code 2>

# English Description 3

<Python Code 3>
```

Each English description/question starts with a '#' and is followed by its corresponding python code. Each data point that we look for comprises of a question and its corresponding python code. We can therefore look for the first charecter in each line to detrmine the start of the next data point. All lines between two lines starting with a '#' form a part of the python solution.

To further parse out the python code we make use of python's source code [tokenizer](https://docs.python.org/3/library/tokenize.html) in order to effectively deal with code syntax and indentation(spaces and tabs). 

## Data Augmentation - Random Variable Replacement

Since we have mere 5000 data points, we make use of data augmentations to increase the size of our dataset. While tokenizing the python code, we mask the names of certain variables randomly(with 'var_1, 'var_2' etc) to ensure that the model that we train does not merly fixate on the way the variables are named and actually tries to understand the inhrent logic and syntax of the python code.

For example consider the folowing program:

```
def add_two_numbers (num1 ,num2 ):
    sum =num1 +num2 
    return sum
```

we can replace some of the above variables to create new data points. The following are valid augmentations:

1. 
```
def add_two_numbers (var_1 ,num2 ):
    sum =var_1 +num2 
    return sum
```
2.
 ```
def add_two_numbers (num1 ,var_1 ):
    sum =num1 +var_1 
    return sum
```
3.
```
def add_two_numbers (var_1 ,var_2 ):
    sum = var_1 + var_2 
    return sum
```

In the above example, we have therefore exapnded a single data point into 3 more data points using our random variable replacement technique.

## Model Archietecture

We will be using the transformer model as explained in this [blog](https://ai.plainenglish.io/lets-pay-attention-to-transformers-a1c2dc566dbd) to perform sequence to sequence learning on our dataset. Here we will be treating the english description/question as our source and the corresponding python code as the target for our training. 

## Loss function -  Cross Entropy with label smoothening

We have used augmentations in our dataset to mask variable literals. This means that our model can predict a variety of values for a particular variable and all of them are correct as long as the predictions are consistent through the code. This would mean that our training labels are not very certain and hence it would make more sense to treat them to be correct with probability 1- smooth_eps and incorrect otherwise. This is what label smoothening does. By adding [label smoothening](https://arxiv.org/abs/1906.02629) to Cross Entropy we ensure that the model does not become too confident on predicting some of our varibles that can be replced via augmentations. 

# Example Outputs



