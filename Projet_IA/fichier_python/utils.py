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
    # Ensemble des exemples de classe -1:
    negatifs = desc[labels == -1]
    # Ensemble des exemples de classe +1:
    positifs = desc[labels == +1]
    test_n = np.random.uniform(0,1,len(negatifs))
    test_p = np.random.uniform(0,1,len(positifs))
    # Affichage de l'ensemble des exemples :
    plt.scatter(test_n,negatifs,marker='o') # 'o' pour la classe -1
    plt.scatter(test_p,positifs,marker='x') # 'x' pour la classe +1

def euclidean_distance(img_a, img_b):
    somme = 0
    for i,j in zip(img_a,img_b):
        somme = somme + (i-j)**2
    return np.sqrt(somme)

def normalisation(DF):
    return (DF - DF.min())/(DF.max()-DF.min())

def majority(labels):
    plus =0
    moins=0
    for i in labels:
        if (i==1):
            plus += 1
        if (i==-1):
            moins += 1
    if(plus >= moins):
        return 1
    if(moins > plus):
        return -1

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