#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:46:21 2019

@author: wangzheng
"""
import random
import time

" Problem 2-A-(b) ----------------------------------------------------------- "
def mean_deviation(l):
    m = len(l)
    mean = sum(l)/m
    square_sum = 0
    for x in l:
        square_sum += (x-mean)**2
    deviation = (1/m*square_sum)**0.5
    return mean, deviation
" --------------------------------------------------------------------------- "

" Problem 2-A-(c) ----------------------------------------------------------- "
house_file = open('housing.txt', 'r')
data = [line.rstrip('\n').split(',') for line in house_file]
house_file.close()

size = [int(i[0]) for i in data]
bedroom = [int(i[1]) for i in data]
price = [i[2] for i in data]

size_median, size_deviation = mean_deviation(size)
bedroom_median, bedroom_deviation = mean_deviation(bedroom)

normalized_size = [str((i-size_median)/size_deviation) for i in size]
normalized_bedroom = [str((i-bedroom_median)/bedroom_deviation) for i in bedroom]

normalized_file = open('normalized.txt', 'w')
for i in range(len(data)):
    newline = normalized_size[i] + ',' + normalized_bedroom[i] + ',' + price[i]
    normalized_file.write(newline + '\n')
normalized_file.close()
" --------------------------------------------------------------------------- "

" Problem 2-B-(b) ----------------------------------------------------------- "
import matplotlib.pyplot as plot

normalizedfile = open('normalized.txt', 'r')
normalized_data = [line.rstrip('\n').split(',') for line in normalizedfile]
normalizedfile.close()

def gradient_descent(data, rate):
    w = [0, 0, 0]
    iter = 0
    plot_data = []
    t0 = time.time()
    while iter <= 80:
        for i in range(3):
            w[i] = w[i] - rate * derivative(data, w, i)
        iter += 1
        if iter in [10, 20, 30, 40, 50, 60, 70, 80]:
            plot_data.append(J_cost(data, w))
    print("Running time for batch gradient descent is: ", time.time() - t0)
    return w, plot_data

def derivative(data, w, index):
    result = 0
    if index == 0:
        for i in range(len(data)):
            result += w[0] + w[1]*float(data[i][0]) + w[2]*float(data[i][1]) - float(data[i][2])
    else:
        for i in range(len(data)):
            result += (w[0] + w[1]*float(data[i][0]) + w[2]*float(data[i][1]) \
                - float(data[i][2])) * float(data[i][index-1])
    return result/len(data)
      
def J_cost(data, w):
    result = 0
    for i in range(len(data)):
        result += (w[0] + w[1]*float(data[i][0]) + w[2]*float(data[i][1]) - float(data[i][2])) ** 2
    return result/len(data)/2

w1, plot_data1 = gradient_descent(normalized_data, 0.01)
w2, plot_data2 = gradient_descent(normalized_data, 0.1)
w3, plot_data3 = gradient_descent(normalized_data, 0.2)

iteration = [10, 20, 30, 40, 50, 60, 70, 80]

fig1 = plot.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(iteration, plot_data1, marker='o')
ax1.set(xlabel='Iterations (alpha = 0.01)', ylabel='J (w)')

fig2 = plot.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(iteration, plot_data2, marker='o')
ax2.set(xlabel='Iterations (alpha = 0.1)', ylabel='J (w)')

fig3 = plot.figure()
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(iteration, plot_data3, marker='o')
ax3.set(xlabel='Iterations (alpha = 0.2)', ylabel='J (w)')
" --------------------------------------------------------------------------- "

" Problem 2-B-(c) ----------------------------------------------------------- "
w4, plot_data4 = gradient_descent(normalized_data, 0.03)
w5, plot_data5 = gradient_descent(normalized_data, 0.5)

fig4 = plot.figure()
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(iteration, plot_data4, marker='o')
ax4.set(xlabel='Iterations (alpha = 0.03)', ylabel='J (w)')

fig5 = plot.figure()
ax5 = fig5.add_subplot(1,1,1)
ax5.plot(iteration, plot_data5, marker='o')
ax5.set(xlabel='Iterations (alpha = 0.5)', ylabel='J (w)')
" --------------------------------------------------------------------------- "

" Problem 2-C --------------------------------------------------------------- "
test = [2650, 4]
normalized_test = [(2650-size_median)/size_deviation, (4-bedroom_median)/bedroom_deviation]

print("The price prediction is: ", w5[0] + w5[1]*normalized_test[0] + w5[2]*normalized_test[1])
" --------------------------------------------------------------------------- "

" Problem 2-D --------------------------------------------------------------- "
def stochastic_gradient_descent(data, rate):
    w = [0, 0, 0]
    passes = 0
    plot_data = []
    t0 = time.time()
    while passes < 3:
        if passes > 0:
            random.shuffle(data)
        for i in range(len(data)):
            for j in range(3):
                if j == 0:
                    w[j] = w[j] - rate*(w[0] + w[1]*float(data[i][0]) + w[2]*float(data[i][1]) \
                     - float(data[i][2]))
                else:
                    w[j] = w[j] - rate*(w[0] + w[1]*float(data[i][0]) + w[2]*float(data[i][1]) \
                     - float(data[i][2]))*float(data[i][j-1])
        passes += 1
        plot_data.append(J_cost(data, w))
        print("Cost function of SGD: ", J_cost(data, w))
    print("Running time for SGD: ", time.time() - t0)
    return w, plot_data


s1, s_plot_data1 = stochastic_gradient_descent(normalized_data, 0.05)






