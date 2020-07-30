#!/usr/bin/env python
# coding: utf-8

# # Natural Language Entity Extraction
# Goal:
# 
# - Extracting disease labels from clinical reports
#     - Text matching
#     - Evaluating a labeler
#     - Negation detection
#     - Dependency parsing
# - Question Answering with BERT
#     - Preprocessing text for input
#     - Extracting answers from model output 


# Import Packages
import matplotlib.pyplot as plt # standard plotting library
import nltk # an NLP package
import pandas as pd # use this to keep track of data
import tensorflow as tf # standard deep learning library
import numpy as np
from transformers import * # convenient access to pretrained natural language models


# Load the helper `util` library in order to abstract some of the details. 
import util
from util import *

# Have a look at the dataset:
print("test_df size: {}".format(test_df.shape))
test_df.head()


# Here are a few example impressions
for i in range(3):
    print(f'\nReport Impression {i}:')
    print(test_df.loc[i, 'Report Impression'])


# Goal will be to extract the presence or absence of different abnormalities from the raw text. 
# Next, see the distribution of abnormalities in the dataset. 
plt.figure(figsize=(12,5))
plt.barh(y=CATEGORIES, width=test_df[CATEGORIES].sum(axis=0)) # placed all the names of these abnormalities in a list named `CATEGORIES` to graph below
plt.show()

#
# Can access list of relevant keywords for each pathology label by calling the `get_mention_keywords(observation)` function.
cat = CATEGORIES[2]
related_keywords = get_mention_keywords(cat)
print("Related keywords for {} are:\n{}".format(cat, ', '.join(related_keywords)))


# Start constructing labels for each report. Fill in the `get_labels()` function below.
# - It takes in a report (as an array of sentences) and returns a dictionary that maps each category to a boolean value.
# Will indicates the presence or absence of the abnormality.
def get_labels(sentence_l):
    """
    Returns a dictionary that indicates presence of each category (from CATEGORIES) 
    in the given sentences.
    hint: loop over the sentences array and use get_mention_keywords() function.
    
    Args: 
        sentence_l (array of strings): array of strings representing impression section
    Returns:
        observation_d (dict): dictionary mapping observation from CATEGORIES array to boolean value
    """
    observation_d = {}
   
    # loop through each category
    for cat in CATEGORIES:
        
        # Initialize the observations for all categories to be False
        observation_d[cat] = False 

    # For each sentence in the list:
    for s in sentence_l: 
        
        # Set the characters to all lowercase, for consistent string matching
        s = s.lower()
        
        # for each category
        for cat in CATEGORIES: 
            
            # for each phrase that is related to the keyword (use the given function)
            for phrase in get_mention_keywords(cat):
            
                # make the phrase all lowercase for consistent string matching
                phrase = phrase.lower()
                
                # check if the phrase appears in the sentence
                if phrase in s: 
                    observation_d[cat] = True

    return observation_d


# In[8]:


print("Test Case")

test_sentences = ["Diffuse Reticular Pattern, which can be seen with an atypical infection or chronic fibrotic change.", 
                  "no Focal Consolidation."]
print("\nTest Sentences:\n")
for s in test_sentences:
    print(s)

print()
retrieved_labels = get_labels(test_sentences)
print("Retrieved labels: ")

for key, value in sorted(retrieved_labels.items(), key=lambda x: x[0]): 
    print("{} : {}".format(key, value))
print()

print("Expected labels: ")
expected_labels = {'Cardiomegaly': False, 'Lung Lesion': False, 'Airspace Opacity': True, 'Edema': False, 'Consolidation': True, 'Pneumonia': True, 'Atelectasis': False, 'Pneumothorax': False, 'Pleural Effusion': False, 'Pleural Other': False, 'Fracture': False}
for key, value in sorted(expected_labels.items(), key=lambda x: x[0]): 
    print("{} : {}".format(key, value))
print()

for category in CATEGORIES:
    if category not in retrieved_labels:
        print(f'Category {category} not found in retrieved labels. Please check code.')
    
    elif retrieved_labels[category] == expected_labels[category]:
        print(f'Labels match for {category}!')
    
    else:
        print(f'Labels mismatch for {category}. Please check code.')


#
# Run `get_f1_table()` below to calculate your function's performance on the whole test set.
# Takes advantage of modules from the `bioc` and `negbio` python packages to intelligently split paragraph to sentences and then apply function on each sentence. 
get_f1_table(get_labels, test_df) 
 
# Run the following example to see how cleanup changes the input. 
raw_text = test_df.loc[28, 'Report Impression']
print("raw text: \n\n" + raw_text)
print("cleaned text: \n\n" + clean(raw_text))

#
get_f1_table(get_labels, test_df, cleanup=True) # can add to the pipeline and see if cleaning the text can improve performance



# Implement your `get_labels()` function one more time.
# - Use a boolean "flag" to indicate whether a negation like "no" or "not" appears in a sentence.
# - Only set a label to `True` if the word "not" or "no" did not appear in the sentence.
def get_labels_negative_aware(sentence_l):
    """
    Returns a dictionary that indicates presence of categories in
    sentences within the impression section of the report.
    Only set a label to True if no 'negative words' appeared in the sentence.
    hint: loop over the sentences array and use get_mention_keywords() function.
    
    Args: 
        sentence_l (array of strings): array of strings representing impression section
    Returns:
        observation_d (dict): dictionary mapping observation from CATEGORIES array to boolean value
    """
    # Notice that all of the negative words are written in lowercase
    negative_word_l = ["no", "not", "doesn't", "does not", "have not", "can not", "can't", "n't"]
    observation_d = {}
    
    # Initialize the observation dictionary 
    # so that all categories are not marked present.
    for cat in CATEGORIES: 
        
        # Initialize category to not present.
        observation_d[cat] = False

    # Loop through each sentence in the list of sentences
    for s in sentence_l: 
        
        # make the sentence all lowercase
        s = s.lower()
        
        # Initialize the flag to indicate no negative mentions (yet)
        negative_flag = False
        
        # Go through all the negative words in the list
        for neg in negative_word_l: 
            
            # Check if the word is a substring in the sentence
            if neg in s: 
                # set the flag to indicate a negative mention
                negative_flag = True
                
                # Once a single negative mention is found,
                # you can stop checking the remaining negative words
                break 

        # When a negative word was not found in the sentence,
        # check for the presence of the diseases
        if not negative_flag: 
            
            # Loop through the categories list
            for cat in CATEGORIES:
                
                # Loop through each phrase that indicates this category
                for phrase in get_mention_keywords(cat): # complete this line

                        # make the phrase all lowercase
                        phrase = phrase.lower()
                        
                        # Check if the phrase is a substring in the sentence
                        if phrase in s: 
                            
                            # Set the observation dictionary
                            # to indicate the presence of this category
                            observation_d[cat] = True
    
    return observation_d


print("Test Case")

test_sentences = ["Diffuse Reticular pattern, which can be seen with an atypical infection or chronic fibrotic change.", 
                  "No Focal Consolidation."]
print("\nTest Sentences:\n")
for s in test_sentences:
    print(s)

print()
retrieved_labels = get_labels_negative_aware(test_sentences)
print("Retrieved labels: ")

for key, value in sorted(retrieved_labels.items(), key=lambda x: x[0]): 
    print("{} : {}".format(key, value))
print()

print("Expected labels: ")
expected_labels = {'Cardiomegaly': False, 'Lung Lesion': False, 'Airspace Opacity': True, 'Edema': False, 'Consolidation': False, 'Pneumonia': True, 'Atelectasis': False, 'Pneumothorax': False, 'Pleural Effusion': False, 'Pleural Other': False, 'Fracture': False}
for key, value in sorted(expected_labels.items(), key=lambda x: x[0]): 
    print("{} : {}".format(key, value))
print()

print("Test Results:")
for category in CATEGORIES:
    if category not in retrieved_labels:
        print(f'Category {category} not found in retrieved labels. Please check code.')
    
    elif retrieved_labels[category] == expected_labels[category]:
        print(f'Labels match for {category}!')
    
    else:
        print(f'Labels mismatch for {category}. Please check code.')


# With the basic labeling method `get_labels()`, this set Consolidation to True, because it didn't look for 'negative' words.
# Check how this changes your aggregate performance:
get_f1_table(get_labels_negative_aware, test_df, cleanup=True)

#
# Can do comparison on a smaller subset of the data (200 samples). 
# Run the following cells to get predictions using the `negbio` engine 
sampled_test = test_df.sample(200,random_state=0)

# Run the next cell to extract predictions from the sampled test set. 
# Note: This should take about **5 minutes** to run.
negbio_preds = get_negbio_preds(sampled_test)


# Next, calculate the new F1 scores to see the dependency parser does. 
calculate_f1(sampled_test, negbio_preds)


# Finally, let's compare all methods side by side!
basic = get_f1_table(get_labels, sampled_test).rename(columns={"F1": "F1 Basic"})
clean_basic = get_f1_table(get_labels, sampled_test, cleanup=True).rename(columns={"F1": "F1 Cleaned"})
negated_basic = get_f1_table(get_labels_negative_aware, sampled_test, cleanup=True).rename(columns={"F1": "F1 Negative Basic"})
negated_negbio = calculate_f1(sampled_test, negbio_preds).rename(columns={"F1": "F1 Negbio"})

joined_preds = basic.merge(clean_basic, on="Label")
joined_preds = joined_preds.merge(negated_basic, on="Label")
joined_preds = joined_preds.merge(negated_negbio,  on="Label")

joined_preds
 
# 
# Use the tokenizer to prepare the input mapping each word to a unique element in the vocabulary and inserting special tokens. 
tokenizer = AutoTokenizer.from_pretrained("./models")


def prepare_bert_input(question, passage, tokenizer, max_seq_length=384):
    """
    Prepare question and passage for input to BERT. 

    Args:
        question (string): question string
        passage (string): passage string where answer should lie
        tokenizer (Tokenizer): used for transforming raw string input
        max_seq_length (int): length of BERT input
    
    Returns:
        input_ids (tf.Tensor): tensor of size (1, max_seq_length) which holds
                               ids of tokens in input
        input_mask (list): list of length max_seq_length of 1s and 0s with 1s
                           in indices corresponding to input tokens, 0s in
                           indices corresponding to padding
        tokens (list): list of length of actual string tokens corresponding to input_ids
    """
    # tokenize question
    question_tokens = tokenizer.tokenize(question)
    
    # tokenize passage
    passage_token = tokenizer.tokenize(passage)

    # get special tokens 
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    
    # manipulate tokens to get input in correct form (not adding padding yet)
    # CLS {question_tokens} SEP {answer_tokens} 
    # This should be a list of tokens
    tokens = [CLS, *question_tokens, SEP, *passage_token]

    
    # Convert tokens into integer IDs.
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Create an input mask which has integer 1 for each token in the 'tokens' list
    input_mask = [1] * len(input_ids)

    # pad input_ids with 0s until it is the max_seq_length
    # Create padding for input_ids by creating a list of zeros [0,0,...0]
    # Add the padding to input_ids so that its length equals max_seq_length
    input_ids = input_ids + ([0] * (max_seq_length - len(input_ids)))
    
    # Do the same to pad the input_mask so its length is max_seq_length
    input_mask = input_mask + ([0] * (max_seq_length - len(input_mask)))


    return tf.expand_dims(tf.convert_to_tensor(input_ids), 0), input_mask, tokens  


# Test by running it on your sample input.
passage = "My name is Bob."

question = "What is my name?"

input_ids, input_mask, tokens = prepare_bert_input(question, passage, tokenizer, 20)
print("Test Case:\n")
print("Passage: {}".format(passage))
print("Question: {}".format(question))
print()
print("Tokens:")
print(tokens)
print("\nCorresponding input IDs:")
print(input_ids)
print("\nMask:")
print(input_mask)


# 
# Fill in the following function to get the best start and end locations given the start and end scores and the input mask. 
def get_span_from_scores(start_scores, end_scores, input_mask, verbose=False):
    """
    Find start and end indices that maximize sum of start score
    and end score, subject to the constraint that start is before end
    and both are valid according to input_mask.

    Args:
        start_scores (list): contains scores for start positions, shape (1, n)
        end_scores (list): constains scores for end positions, shape (1, n)
        input_mask (list): 1 for valid positions and 0 otherwise
    """
    n = len(start_scores)
    max_start_i = -1
    max_end_j = -1
    max_start_score = -np.inf
    max_end_score = -np.inf
    max_sum = -np.inf
    
    # Find i and j that maximizes start_scores[i] + end_scores[j]
    # so that i <= j and input_mask[i] == input_mask[j] == 1
    
    # set the range for i
    for i in range(n): 
        
        # set the range for j
        for j in range(i, n): 

            # both input masks should be 1
            if input_mask[i] == input_mask[j] == 1: 
                
                # check if the sum of the start and end scores is greater than the previous max sum
                if (start_scores[i] + end_scores[j]) > max_sum: 

                    # calculate the new max sum
                    max_sum = start_scores[i] + end_scores[j]
        
                    # save the index of the max start score
                    max_start_i = i
                
                    # save the index for the max end score
                    max_end_j = j
                    
                    # save the value of the max start score
                    max_start_val = start_scores[i]
                    
                    # save the value of the max end score
                    max_end_val = end_scores[j]

    if verbose:
        print(f"max start is at index i={max_start_i} and score {max_start_val}")
        print(f"max end is at index i={max_end_j} and score {max_end_val}")
        print(f"max start + max end sum of scores is {max_sum}")
    return max_start_i, max_end_j


# Test this out on the following sample start scores and end scores:
start_scores = tf.convert_to_tensor([-1, 2, 0.4, -0.3, 0, 8, 10, 12], dtype=float)
end_scores = tf.convert_to_tensor([5, 1, 1, 3, 4, 10, 10, 10], dtype=float)
input_mask = [1, 1, 1, 1, 1, 0, 0, 0]

start, end = get_span_from_scores(start_scores, end_scores, input_mask, verbose=True)

print("Expected: (1, 4) \nReturned: ({}, {})".format(start, end))


# Test 2
start_scores = tf.convert_to_tensor([0, 2, -1, 0.4, -0.3, 0, 8, 10, 12], dtype=float)
end_scores = tf.convert_to_tensor([0, 5, 1, 1, 3, 4, 10, 10, 10], dtype=float)
input_mask = [1, 1, 1, 1, 1, 0, 0, 0, 0 ]

start, end = get_span_from_scores(start_scores, end_scores, input_mask, verbose=True)

print("Expected: (1, 1) \nReturned: ({}, {})".format(start, end))

#
# Add some post-processing to get the final string. 
def construct_answer(tokens):
    """
    Combine tokens into a string, remove some hash symbols, and leading/trailing whitespace.
    Args:
        tokens: a list of tokens (strings)
    
    Returns:
        out_string: the processed string.
    """
    
    # join the tokens together with whitespace
    out_string = " ".join(tokens)
    
    # replace ' ##' with empty string
    out_string = re.sub(' ##', '', out_string)
    
    # remove leading and trailing whitespace
    out_string = out_string.strip()

    # if there is an '@' symbol in the tokens, remove all whitespace
    if '@' in tokens:
        out_string = out_string.replace(' ', '')

    return out_string


# Test

tmp_tokens_1 = [' ## hello', 'how ', 'are ', 'you?      ']
tmp_out_string_1 = construct_answer(tmp_tokens_1)

print(f"tmp_out_string_1: {tmp_out_string_1}, length {len(tmp_out_string_1)}")


tmp_tokens_2 = ['@',' ## hello', 'how ', 'are ', 'you?      ']
tmp_out_string_2 = construct_answer(tmp_tokens_2)
print(f"tmp_out_string_2: {tmp_out_string_2}, length {len(tmp_out_string_2)}")

#
# Now, load the pre-trained model
model = TFAutoModelForQuestionAnswering.from_pretrained("./models") # takes all the functions implemented and performs question-answering


# Using the helper functionsimplemented andfill out the get_model_answer function to create question-answering system.
def get_model_answer(model, question, passage, tokenizer, max_seq_length=384):
    """
    Identify answer in passage for a given question using BERT. 

    Args:
        model (Model): pretrained Bert model which we'll use to answer questions
        question (string): question string
        passage (string): passage string
        tokenizer (Tokenizer): used for preprocessing of input
        max_seq_length (int): length of input for model
        
    Returns:
        answer (string): answer to input question according to model
    """ 
    # prepare input: use the function prepare_bert_input
    input_ids, input_mask, tokens = prepare_bert_input(question, passage, tokenizer, max_seq_length)
    
    # get scores for start of answer and end of answer
    # use the model returned by TFAutoModelForQuestionAnswering.from_pretrained("./models")
    # pass in in the input ids that are returned by prepare_bert_input
    start_scores, end_scores = model(input_ids)
    
    # start_scores and end_scores will be tensors of shape [1,max_seq_length]
    # To pass these into get_span_from_scores function, 
    # take the value at index 0 to get a tensor of shape [max_seq_length]
    start_scores = start_scores[0]
    end_scores = end_scores[0]
    
    # using scores, get most likely answer
    # use the get_span_from_scores function
    span_start, span_end = get_span_from_scores(start_scores, end_scores, input_mask)
    
    # Using array indexing to get the tokens from the span start to span end (including the span_end)
    answer_tokens = tokens[span_start:span_end+1]
    
    # Combine the tokens into a single string and perform post-processing
    # use construct_answer
    answer = construct_answer(answer_tokens)
    
    return answer


# Now all the pieces are prepared, let's try an example from the SQuAD dataset. 
passage = "Computational complexity theory is a branch of the theory            of computation in theoretical computer science that focuses            on classifying computational problems according to their inherent            difficulty, and relating those classes to each other. A computational            problem is understood to be a task that is in principle amenable to            being solved by a computer, which is equivalent to stating that the            problem may be solved by mechanical application of mathematical steps,            such as an algorithm."

question = "What branch of theoretical computer science deals with broadly             classifying computational problems by difficulty and class of relationship?"

print("Output: {}".format(get_model_answer(model, question, passage, tokenizer)))
print("Expected: Computational complexity theory")


# 

passage = "The word pharmacy is derived from its root word pharma which was a term used since            the 15th–17th centuries. However, the original Greek roots from pharmakos imply sorcery            or even poison. In addition to pharma responsibilities, the pharma offered general medical            advice and a range of services that are now performed solely by other specialist practitioners,            such as surgery and midwifery. The pharma (as it was referred to) often operated through a            retail shop which, in addition to ingredients for medicines, sold tobacco and patent medicines.            Often the place that did this was called an apothecary and several languages have this as the            dominant term, though their practices are more akin to a modern pharmacy, in English the term            apothecary would today be seen as outdated or only approproriate if herbal remedies were on offer            to a large extent. The pharmas also used many other herbs not listed. The Greek word Pharmakeia            (Greek: φαρμακεία) derives from pharmakon (φάρμακον), meaning 'drug', 'medicine' (or 'poison')."

question = "What word is the word pharmacy taken from?"

print("Output: {}".format(get_model_answer(model, question, passage, tokenizer)))
print("Expected: pharma")


# Now, try it on clinical notes. Below is an excerpt of a doctor's notes for a patient with an abnormal echocardiogram.
passage = "Abnormal echocardiogram findings and followup. Shortness of breath, congestive heart failure,            and valvular insufficiency. The patient complains of shortness of breath, which is worsening.            The patient underwent an echocardiogram, which shows severe mitral regurgitation and also large            pleural effusion. The patient is an 86-year-old female admitted for evaluation of abdominal pain            and bloody stools. The patient has colitis and also diverticulitis, undergoing treatment.            During the hospitalization, the patient complains of shortness of breath, which is worsening.            The patient underwent an echocardiogram, which shows severe mitral regurgitation and also large            pleural effusion. This consultation is for further evaluation in this regard. As per the patient,            she is an 86-year-old female, has limited activity level. She has been having shortness of breath            for many years. She also was told that she has a heart murmur, which was not followed through            on a regular basis."

q1 = "How old is the patient?"
q2 = "Does the patient have any complaints?"
q3 = "What is the reason for this consultation?"
q4 = "What does her echocardiogram show?"
q5 = "What other symptoms does the patient have?"


questions = [q1, q2, q3, q4, q5]

for i, q in enumerate(questions):
    print("Question {}: {}".format(i+1, q))
    print()
    print("Answer: {}".format(get_model_answer(model, q, passage, tokenizer)))
    print()
    print()
