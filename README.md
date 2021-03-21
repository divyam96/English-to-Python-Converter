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

.
.
.
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

## Model Architecture
![Transformer](/res/transformer_multihead.png)

We will be using the transformer model as explained in this [blog](https://ai.plainenglish.io/lets-pay-attention-to-transformers-a1c2dc566dbd) to perform sequence to sequence learning on our dataset. Here we will be treating the english description/question as our source(SRC) and the corresponding python code as the target(TRG) for our training. 

### Tokenizing SRC and TRG sequences

We use spacy's default tokenizer to tokenize our SRC sequence.
```
SRC = [' ', 'write', 'a', 'python', 'function', 'to', 'add', 'two', 'user', 'provided', 'numbers', 'and', 'return', 'the', 'sum']
```

We use python's source code [tokenizer](https://docs.python.org/3/library/tokenize.html) to tokenize our TRG. Python's tokenizer returns several attributes for each token. We only extract the token type and the corresponding string attribute in form of a tuple(i.e., (token_type_int, token_string)) as the final token. Our TRG is a sequence of such tuples.
```
TRG = [(57, 'utf-8'), (1, 'def'), (1, 'add_two_numbers'), (53, '('), (1, 'num1'), (53, ','), (1, 'var_1'), (53, ')'), (53, ':'), (4, '\n'), (5, '    '), (1, 'sum'), (53, '='), (1, 'num1'), (53, '+'), (1, 'var_1'), (4, '\n'), (1, 'return'), (1, 'sum'), (4, ''), (6, ''), (0, '')]
```

## Loss function -  Cross Entropy with label smoothening

We have used augmentations in our dataset to mask variable literals. This means that our model can predict a variety of values for a particular variable and all of them are correct as long as the predictions are consistent through the code. This would mean that our training labels are not very certain and hence it would make more sense to treat them to be correct with probability 1- smooth_eps and incorrect otherwise. This is what label smoothening does. By adding [label smoothening](https://arxiv.org/abs/1906.02629) to Cross Entropy we ensure that the model does not become too confident on predicting some of our varibles that can be replced via augmentations. 

We use the validation loss and training loss to determine when our model is trained. The model with minimum validation loss is used as the final trained model. 

It is important to note that label smoothening leads to much higher loss values as compared to models that do not make use of label smoothening. But this is as expected as we do not intetend to be certain with our label predictions. This is particularly the case with variables as there can be multiple correct options as long as the predictions are consistent through the target code sequence. 

# Sample Multi-Head Attention Map
![Attention Map for a function that adds two numbers](/res/attention_python_code_generator.png)

# Example Outputs

### Example 1
**Input:**

"program to sort a list of dictionaries by key"

**Output:**
```
var_1 ={'Nikhil':{'roll':24 ,'marks':17 },
'Akshat':{'roll':54 ,'marks':12 },
'Akash':{'roll':15 },'marks':15 }}
sort_key ='marks'
res ='marks'
res =var_2 (test_dict .items (),key =lambda x :x [1 ][sort_key ])
print ("The sorted dictionary by marks is : "+str (res ))
```
### Example 2
**Input:**

"function to merge two lists"

**Output:**
```
def merge_lists (l1 :list ,var_1 :list ):
    return list (zip (l1 ,var_1 ))
```
### Example 3
**Input:**

"program to find gcd"

**Output:**
```
def compute_hcf (x ,y ):

   while (y ):
       x ,y =y ,x %y 
   return x 
```
### Example 4
**Input:**

"program to calculate simple interest"

**Output:**
```
princ_amount =float (input (" Please Enter the Principal Amount : "))
var_1 =float (input (" Please Enter the Rate Of Interest   : "))
var_2 =float (input (" Please Enter Time period in Years   : "))

simple_interest =(princ_amount *var_1 *var_2 )/100 

print ("\nSimple Interest for Principal Amount {0} = {1}".format (princ_amount ,simple_interest ))
```
### Example 5
**Input:**

"function to sum odd elements of list"

**Output:**
```
def sum_odd_elements (l :list ):
    return sum ([i for i in l if i %2 ==1 ])
```
### Example 6
**Input:**

"program to multiply integers in a list"

**Output:**
```
def multiplyList (myList ):
    result =1 
    for x in myList :
         result =result *x 
    return result 
```
### Example 7
**Input:**

"program to reverse a string"

**Output:**
```
var_1 ='Today is bad day'
var_1 [::-1 ]
```
### Example 8
**Input:**

"function to find reverse of a string"

**Output:**
```
def revStr (inp ):
  inp =inp [::-1 ]
  return inp 
```

### Example 9
**Input:**

"program to find the length of tuple"

**Output:**
```
thistuple =("apple","banana","cherry")
print (len (thistuple ))
```

### Example 10
**Input:**

"program to find the area of a square"

**Output:**
```
side =5 
print (f'Area: {side * side}')
```

### Example 11
**Input:**

"program to print epoch timestamp"

**Output:**
```
import time 
import datetime 
str1 ="20/01/2020"
element =datetime .strptime (str1 ,"%d/%m/%Y")
timestamp =datetime .timestamp (element )
print (timestamp )
```

### Example 12
**Input:**

"program to find ascii value of charecter"

**Output:**
```
var_1 ='5'
print (ord (var_1 ))
```

### Example 13
**Input:**

"function to find fibonacci sequence"

**Output:**
```
def recur_fibo (n ):
   if n <=1 :
       return n 
   else :
       return (recur_fibo (n -1 )+recur_fibo (n -2 ))
```

### Example 14
**Input:**

"function to find largest element in a dictionary"

**Output:**
```
def dic_largest (var_1 ):
    return max (var_1 .values ())
```

### Example 15
**Input:**

"program to convert list of tuples into a dictionary"

**Output:**
```
var_1 =[("x",1 ),("x",2 ),("y",1 ),("y",1 ),("y",2 ),("z",1 )]
d ={}
for a ,var_2 in var_1 :
    d .setdefault (a ,[]).append (var_2 )
print (d )
```

### Example 16
**Input:**

"function to convert to binary"

**Output:**
```
def convertToBinary (n ):
   if n >1 :
       convertToBinary (n //2 )
   print (n %2 ,end ='')
```
### Example 17
**Input:**

"program to implement a linked list"

**Output:**
```
class Node :

    def __init__ (self ,data ):
        self .data =data 
        self .next =None 

class LinkedList :

    def __init__ (self ):
        self .head =None 

    def reverse (self ):
        prev =None 
        current =self .next 
        while (current is not None ):
            next =current .next 
            current =current =current 
            current =current 
            current =current 
        self .next =current 

    def push (self ,new_data ):
        next =Node (new_data )
        self .next =self .next =self .next 

    def printList (self ):
        temp =self .head =temp 
        while (temp ):
            print (temp )
            temp =temp =temp .next 


llist =LinkedList ()
llist .push (20 )
llist .push (4 )
llist .push (15 )
llist .push (85 )

print ("Given Linked List")
llist .printList ()
llist .printList ()
llist .printList ()
llist .printList ("\nReversed Linked List")
llist .printList ()
```
### Example 18
**Input:**

"program to add lists elementwise"

**Output:**
```
var_1 =[1 ,2 ,3 ,4 ,5 ]
list2 =[5 ,6 ,7 ]
final =[a +b for a in var_1 for b in var_1 ]
print (f"sum of every pair of numbers from two lists:{final}")
```
### Example 19
**Input:**

"program to find common values between two sets"

**Output:**
```
var_1 ={10 ,20 ,30 ,40 ,50 }
set2 ={60 ,70 ,80 ,90 ,10 }
if var_1 ==1 :
    print ("Two sets have no items in common")
else :
  print ("Two sets have items in common")
  print (var_1 .intersection (y ))
```
### Example 20
**Input:**

"program to find number of unique values in a list"

**Output:**
```
input_list =[1 ,2 ,2 ,5 ,4 ,4 ,3 ,8 ]

var_1 =[]

for var_2 in input_list :
    if var_2 not in var_1 :
        var_1 +=1 

print ("No of unique items are:",var_1 )
```
### Example 21
**Input:**

"function to remove empty lists from a list of lists"

**Output:**
```
def removeEmptyList (li ):
  res =[ele for ele in li if ele !=[]]
  return res
```
### Example 22
**Input:**

"write a function to capitalize a string"

**Output:**
```
def capitalize (text ):
    return text .title ()
```
### Example 23
**Input:**

"write a function to find the area of a circle"

**Output:**
```
def circle_area (r ):
    return 22 /7 *r **2
```
### Example 24
**Input:**

"write a python program to merge two dictionaries"

**Output:**
```
x ={'key1':'val1','key2':'val2'}
y ={'key3':'val3','key4':'val4'}
z ={**x ,**y }# z = x | y
```
### Example 25
**Input:**

"write a function to find factorial"

**Output:**
```
def factorial (n ):
    if n ==0 :
        return 1 
    else :
        return n *factorial (n -1 )
```
