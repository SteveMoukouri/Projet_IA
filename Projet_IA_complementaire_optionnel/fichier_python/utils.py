# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ------------------------ 
def afficheImage(img):
    plt.imshow(img.reshape(28,28), cmap='gray')


def plot2DSet(desc,labels):
    """ ndarray * ndarray -> affichage
    """
    label_0 = desc[labels == 0]
    label_1 = desc[labels == 1]
    label_2 = desc[labels == 2]
    label_3 = desc[labels == 3]
    label_4 = desc[labels == 4]
    label_5 = desc[labels == 5]
    label_6 = desc[labels == 6]
    label_7 = desc[labels == 7]
    label_8 = desc[labels == 8]
    label_9 = desc[labels == 9]
    test_0 = np.random.uniform(0,1,len(label_0))
    test_1 = np.random.uniform(0,1,len(label_1))
    test_2 = np.random.uniform(0,1,len(label_2))
    test_3 = np.random.uniform(0,1,len(label_3))
    test_4 = np.random.uniform(0,1,len(label_4))
    test_5 = np.random.uniform(0,1,len(label_5))
    test_6 = np.random.uniform(0,1,len(label_6))
    test_7 = np.random.uniform(0,1,len(label_7))
    test_8 = np.random.uniform(0,1,len(label_8))
    test_9 = np.random.uniform(0,1,len(label_9))
    # Affichage de l'ensemble des exemples :
    plt.scatter(test_0,label_0,marker='o')
    plt.scatter(test_1,label_1,marker='x')
    plt.scatter(test_2,label_2,marker='1')
    plt.scatter(test_3,label_3,marker='s')
    plt.scatter(test_4,label_4,marker='P')
    plt.scatter(test_5,label_5,marker='+')
    plt.scatter(test_6,label_6,marker='2')
    plt.scatter(test_7,label_7,marker='D')
    plt.scatter(test_8,label_8,marker='<')
    plt.scatter(test_9,label_9,marker='>')


def euclidean_distance(img_a, img_b):
    somme = 0
    for i,j in zip(img_a,img_b):
        somme = somme + (i-j)**2
    return np.sqrt(somme)

def normalisation(DF):
    return (DF - DF.min())/(DF.max()-DF.min())

def majority(data_labels):
	label_0 = 0
	label_1 = 0
	label_2 = 0
	label_3 = 0
	label_4 = 0
	label_5 = 0
	label_6 = 0
	label_7 = 0
	label_8 = 0
	label_9 = 0
	for i in data_labels:
		if i==0:
			label_0 += 1
		if i==1:
			label_1 += 1
		if i==2:
			label_2 += 1
		if i==3:
			label_3 += 1
		if i==4:
			label_4 += 1
		if i==5:
			label_5 += 1
		if i==6:
			label_6 += 1
		if i==7:
			label_7 += 1
		if i==8:
			label_8 += 1
		if i==9:
			label_9 += 1
		max_label,label = max([[label_0,0],[label_1,1],[label_2,2],[label_3,3],[label_4,4],[label_5,5],[label_6,6],[label_7,7],[label_8,8],[label_9,9]])
		return label

def for_the_plot_2D_origine_0(desc,labels,dimension):
    array_zero = np.zeros(dimension)
    list_distance = [(euclidean_distance(image, array_zero), label) for (image,label) in zip(desc,labels)]
    list_distance_data = [dist for (dist,_) in list_distance]
    list_distance_data_array = np.array(list_distance_data)
    list_distance_data_array_norm = normalisation(list_distance_data_array)

    list_label = [label for (_,label) in list_distance]
    list_label_array = np.array(list_label)
    return list_distance_data_array_norm,list_label_array





# Class pour Kernel

class KernelPoly:
    def transform(self,data_set,dimension):
        a_return = [0] *  dimension
        index = 0
        for i in range(len(data_set)):
            a_return[index] = data_set[i]
            index += 1
            for j in range(i, len(data_set)):
                a_return[index] = data_set[i] * data_set[j]
                index += 1
        return a_return
