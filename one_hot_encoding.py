import re
# function to get unique values
def unique(list1):
     
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list
# define input string
data = 'mov ebp, esp. push ebp. sub esp 0x4'
print("Input data used :")
print(data)
# cleaning data, and tokenizing
cleaned_data = re.sub('[ ,.]', ' ', data)
tokenized_data = cleaned_data.split()
# define universe of possible input values and sort ascending

universe = unique(tokenized_data)
universe.sort()
print("The universe of words:")
print(universe)
# defining integer encoding using the index of words in universe.
for word in universe:
    print("integer encoding for {} : {}".format(word,universe.index(word)))

integer_encoded_data = []
for w in cleaned_data.split():
    integer_encoded_data.append(universe.index(w))
print("Integer encoded representation of input data : {}".format(integer_encoded_data))

# to get one-hot encoded representation of each word in our data
# we will create an array with a length of universe
# each word will be represented as one at the given index
columns = len(universe)
rows = len(cleaned_data.split())
one_hot_encoding = [[0 for x in range(columns)] for x in range(rows)]
print("one hot encoding representation of input data: ")
for i,k in enumerate(integer_encoded_data):
    one_hot_encoding[i][k]=1
print(one_hot_encoding)

import pandas as pd
print(pd.DataFrame(one_hot_encoding))
"""
output :
Input data used :
mov ebp, esp. push ebp.
The universe of words:
['ebp', 'esp', 'mov', 'push']
integer encoding for ebp is 0
integer encoding for esp is 1
integer encoding for mov is 2
integer encoding for push is 3
Integer encoded representation of input data : [2, 0, 1, 3, 0]
one hot encoding representation of input data: 
[[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
"""