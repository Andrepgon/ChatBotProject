import numpy as np
import re

from flatbuffers.packer import float64

from dialogs_prep import pairs

input_docs = []
target_docs = []

input_set = set()
target_set = set()

for line in pairs:
    input_doc, target_doc = line[0], line[1] #question on input and answer at output
    input_docs.append(input_doc)

    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
    target_doc = "<START> " + target_doc + " <END>"
    target_docs.append(target_doc)

    for token in re.findall(r"[\w']+|[^\s\w]", input_doc): #adding all words from the question sentence to our question dictionary
        input_set.add(token)

    for token in target_doc.split(): #adding all words from the answer sentence to our answer dictionary
        target_set.add(token)


input_set = sorted(input_set)
target_set = sorted(target_set)

#Creating variables to save the length of the dictionaries

num_encoder_set = len(input_set)
num_decoder_set = len(target_set)

#Creating features dictionary and inverse features dictionary for question and answer

max_len_input_sen = max([len(re.findall(r"[\w']+|[^\s\w]", sentence)) for sentence in input_docs])
max_len_target_sen = max([len(re.findall(r"[\w']+|[^\s\w]", sentence)) for sentence in target_docs])

input_features_dict = dict([(word, i) for word,i in enumerate(input_set)])
target_features_dict = dict([(word, i) for word, i in enumerate(target_set)])

reverse_input_features_dict = dict([(i,word) for i,word in input_features_dict.items()])
reverse_target_features_dict = dict([(i,word) for i,word in target_features_dict.items()])

#creating the matrices
encoder_matrix = np.zeros((len(input_docs), max_len_input_sen, num_encoder_set), dtype = 'float32')

decoder_input_matrix = np.zeros((len(input_docs), max_len_target_sen, num_decoder_set), dtype = 'float32')
decoder_target_matrix = np.zeros((len(input_docs), max_len_target_sen, num_decoder_set), dtype = 'float32')

#filling the matrices
for line, (input_sentence, target_sentence) in enumerate(zip(input_docs, target_docs)):

    for timestep, word in enumerate(re.findall(r"[\w']+|[^\s\w]", input_sentence)):
        encoder_matrix[line, timestep, input_features_dict[word]] = 1

    for timestep, word in enumerate(target_sentence.split()):
        decoder_input_matrix[line, timestep, target_features_dict[word]] = 1

        if timestep > 0:
            decoder_target_matrix[line, timestep-1, target_features_dict[word]] = 1








