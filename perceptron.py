#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 20:13:21 2019

@author: wangzheng
"""

" Problem 1 ----------------------------------------------------------------- "
def building_file():
    " reading emails "
    training_file = open('spam_train.txt','r')
    all_emails = []
    for email in training_file:
        all_emails.append(email.rstrip('\n'))
    training_file.close()
    
    validation_emails = all_emails[0:1000]
    training_emails = all_emails[1000:]
    
    del training_emails[107]
    del training_emails[3183]
    del training_emails[3680]
    
    " creating validation.txt and train.txt "
    validation_txt = open('validation.txt','w')
    for email in validation_emails:
        validation_txt.write(email+'\n')
    validation_txt.close()
    
    train_txt = open('train.txt','w')
    for email in training_emails:
        train_txt.write(email+'\n')
    train_txt.close()


" build the vocabulary list "
def words(data, X):
    data_file = open(data, 'r')
    email_list = [email.rstrip('\n').split(' ')[1:] for email in data_file]
    word_dic = {}
    vocabulary = []
    for email in email_list:
        for word in set(email):
            if word not in word_dic:
                word_dic[word] = 1
            else:
                word_dic[word] += 1
    for word in word_dic:
        if word_dic[word] >= X:
            vocabulary.append(word)
    return vocabulary


def feature_vector(email):
    v = [0]*len(vocabulary_list)
    for i in range(len(vocabulary_list)):
        if vocabulary_list[i] in email:
            v[i] = 1
    return v
" --------------------------------------------------------------------------- "


def dot(v1, v2):
    result = 0
    for i in range(len(v1)):
        result += v1[i]*v2[i]
    return result

def scalar_m(v, r):
    result = []
    for i in range(len(v)):
        result.append(v[i]*r)
    return result

def addition(v1, v2):
    result = []
    for i in range(len(v1)):
        result.append(v1[i]+v2[i])
    return result


" Problem 2 ----------------------------------------------------------------- "
def perceptron_train(data):
    w = [0] * len(vocabulary_list)
    k = 0
    iter = 0
    checked = False
    n = len(data)
    print('\nStart training...')
    " Problem 7 ------------------------------------------------------------- "
    while not checked:
#    while iter < 5 and not checked:
        checked = True
        for i in range(n):
            if data[i][0]*dot(w, data[i][1]) > 0 or data[i][1] == [0]*len(vocabulary_list):
                continue
            else:
                w = addition(w, scalar_m(data[i][1],data[i][0]))
                k += 1
                checked = False
        iter += 1
        print('Iterations: ', iter)
    print('Training Completed!')
    print('\nClassification vector[:15]: ', w[:15], 'Updates: ', k, 'Iterations: ', iter)
    return (w, k, iter)


def perceptron_error(w, data):
    n = len(data)
    error = 0
    for i in range(n):
        if data[i][0] * dot(data[i][1], w) <= 0:
            error += 1
    return error/n
" --------------------------------------------------------------------------- "
building_file()
vocabulary_list = words('train.txt', 26)


" constructing training data "
print('Constructing data sets...')
training_file = open('train.txt','r')
training_data = []
for email in training_file:
    training_data.append(email.rstrip('\n').split(' '))
training_file.close()
for i in range(len(training_data)):
    if training_data[i][0] == '0':
        training_data[i][0] = '-1'
    training_data[i] = [int(training_data[i][0]), feature_vector(training_data[i])]

   
" constructing validation data "
vali_file = open('validation.txt','r')
validation_data = []
for email in vali_file:
    validation_data.append(email.rstrip('\n').split(' '))
vali_file.close()
for i in range(len(validation_data)):
    if validation_data[i][0] == '0':
        validation_data[i][0] = '-1'
    validation_data[i] = [int(validation_data[i][0]), feature_vector(validation_data[i])]


" Problem 3 ----------------------------------------------------------------- "
result = perceptron_train(training_data)
print('\nError rate of training data: ', perceptron_error(result[0], training_data))
print('Error rate of validation data: ', perceptron_error(result[0], validation_data))
" --------------------------------------------------------------------------- "


" Problem 4 ----------------------------------------------------------------- "
w = result[0]
w_copy = []
for i in w:
    w_copy.append(i)

positive = []
for i in range(12):
    for x in range(len(w_copy)):
        if w_copy[x] != '0':
            max1 = x
            break
    for j in range(max1, len(w_copy)):
        if w_copy[j] != '0' and w_copy[j] > w_copy[max1]:
            max1 = j
    positive.append(max1)
    w_copy[max1] = '0'


positive_words = [vocabulary_list[i] for i in positive]
print('\n12 words with the most positive weights: ', positive_words)

w_copy = []
for i in w:
    w_copy.append(i)

negative = []
for i in range(12):
    for x in range(len(w_copy)):
        if w_copy[x] != '0':
            min1 = x
            break
    for j in range(min1, len(w_copy)):
        if w_copy[j] != '0' and w_copy[j] < w_copy[min1]:
            min1 = j
    negative.append(min1)
    w_copy[min1] = '0'

negative_words = [vocabulary_list[i] for i in negative]
print('12 words with the most negative weights: ', negative_words)
" --------------------------------------------------------------------------- "


" Problem 5 and 6 ----------------------------------------------------------- "
import matplotlib.pyplot as plot

N = [200, 600, 1200, 2400, 3997]
result_set = [perceptron_train(training_data[:n]) for n in N]
w_set = [i[0] for i in result_set]
error_set = [perceptron_error(w, validation_data) for w in w_set]
iteration_set = [i[2] for i in result_set]

fig1 = plot.figure()
ax = fig1.add_subplot(1,1,1)
ax.plot(N, error_set, marker='o')
ax.set(xlabel='Amount of training data N', ylabel='Error rate')

fig2 = plot.figure()
bx = fig2.add_subplot(1,1,1)
bx.plot(N, iteration_set, marker='o')
bx.set(xlabel='Amount of training data N', ylabel='Number of iterations')
" --------------------------------------------------------------------------- "


" Problem 8 ----------------------------------------------------------------- "
"""
X = 26, limit = 16*   Error rate: 0.023
X = 26, limit = 10    Error rate: 0.022
X = 26, limit = 5     Error rate: 0.026

X = 28, limit = 17*   Error rate: 0.018
X = 28, limit = 12    Error rate: 0.02
X = 28, limit = 8     Error rate: 0.022
X = 28, limit = 6     Error rate: 0.017
X = 28, limit = 5     Error rate: 0.014
X = 28, limit = 3     Error rate: 0.02

X = 22, limit = 10*   Error rate: 0.013 * picked up
X = 22, limit = 7     Error rate: 0.014
X = 22, limit = 5     Error rate: 0.014
"""



#" constructing all training data "
#all_training_data = validation_data + training_data
#
#" constructing test data "
#test_file = open('spam_test.txt','r')
#test_data = []
#for email in test_file:
#    test_data.append(email.rstrip('\n').split(' '))
#test_file.close()
#for i in range(len(test_data)):
#    if test_data[i][0] == '0':
#        test_data[i][0] = '-1'
#    test_data[i] = [int(test_data[i][0]), feature_vector(test_data[i])]
#
#new_result = perceptron_train(all_training_data)
#print('\nError rate of training data: ', perceptron_error(new_result[0], all_training_data))
#print('Error rate of test data: ', perceptron_error(new_result[0], test_data))
" --------------------------------------------------------------------------- "


" Problem 9 ----------------------------------------------------------------- "
#vocabulary_list = words('train.txt', 1500)
#print(vocabulary_list)
#print(len(vocabulary_list))   # 37
#
#result = perceptron_train(training_data)
#print('\nError rate of training data: ', perceptron_error(result[0], training_data))
#print('Error rate of validation data: ', perceptron_error(result[0], validation_data))
" --------------------------------------------------------------------------- "



