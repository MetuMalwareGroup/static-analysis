import numpy as np
import re
def unique(input_list):
    # insert the list to the set
    list_set = set(input_list)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list

data = ['mov ebp, esp. push ebp. sub esp 0x4.', 
        'mov ebp, esp.']
sentence_len = len(data)
# cleaning data, and tokenizing
clean_data = lambda x : re.sub('[,.]', '', x)
cleaned_data = [clean_data(i) for i in data]
print(cleaned_data)
tokenize_data = lambda x : x.split()
tokenized_data = np.hstack(np.array([tokenize_data(i) for i in cleaned_data]))
print(tokenized_data)
vocabulary = unique(tokenized_data)
vocabulary.sort()
print("The universe of words:")
print(vocabulary)
vocabulary_size = len(vocabulary)
print("vocabulary size : {}".format(vocabulary_size) )
print("sentence count : {}".format(sentence_len))  

vector = [[0] * vocabulary_size for i in range(sentence_len)]

values ={ vocabulary.index(word):word for word in vocabulary} 
print(values)
for i,instruction in enumerate(cleaned_data):
    words = instruction.split(' ')
    for word in words:
        vector[i][vocabulary.index(word)]+=1

arr = np.array(vector)
for i in range(sentence_len):
    print(vector[i])

