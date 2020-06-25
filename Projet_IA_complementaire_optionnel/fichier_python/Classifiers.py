# Import de packages externes
import numpy as np
import math as mt
import pandas as pd
import random as random
from fichier_python import utils as ut

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système 
        """
        res = sum([1 for i in range(len(desc_set)) if self.predict(desc_set[i]) == label_set[i]])
        return res/len(label_set)
    
# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.w = np.random.uniform(-1,1,input_dimension)
        self.input_dimension = input_dimension
        
    def train(self, desc_set, label_set):
       	""" Permet d'entrainer le modele sur l'ensemble donné
        """     
        pass
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return sum(self.w * x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0 :
            return -1
        return 1

# ---------------------------

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension,learning_rate):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.epsilon = learning_rate
        self.test_0 = [0] * self.input_dimension
        self.test_1 = [0] * self.input_dimension
        self.test_2 = [0] * self.input_dimension
        self.test_3 = [0] * self.input_dimension
        self.test_4 = [0] * self.input_dimension
        self.test_5 = [0] * self.input_dimension
        self.test_6 = [0] * self.input_dimension
        self.test_7 = [0] * self.input_dimension
        self.test_8 = [0] * self.input_dimension
        self.test_9 = [0] * self.input_dimension

    def train(self, desc_set, label_set,nb_iteration):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        for j in range(nb_iteration):
        	for i in range(len(desc_set)):
        		if 0 == label_set[i]:
        			self.test_0 = self.test_0 + (desc_set[i] * self.epsilon)

        		if 1 == label_set[i]:
        			self.test_1 = self.test_1 + (desc_set[i] * self.epsilon)

        		if 2 == label_set[i]:
        			self.test_2 = self.test_2 + (desc_set[i] * self.epsilon)

        		if 3 == label_set[i]:
        			self.test_3 = self.test_3 + (desc_set[i] * self.epsilon)

        		if 4 == label_set[i]:
        			self.test_4 = self.test_4 + (desc_set[i] * self.epsilon)

        		if 5 == label_set[i]:
        			self.test_5 = self.test_5 + (desc_set[i] * self.epsilon)

        		if 6 == label_set[i]:
        			self.test_6 = self.test_6 + (desc_set[i] * self.epsilon)

        		if 7 == label_set[i]:
        			self.test_7 = self.test_7 + (desc_set[i] * self.epsilon)

        		if 8 == label_set[i]:
        			self.test_8 = self.test_8 + (desc_set[i] * self.epsilon)

        		if 9 == label_set[i]:
        			self.test_9 = self.test_9 + (desc_set[i] * self.epsilon)


    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        maxi_sum,label = max([[sum(self.test_0 * x),0],[sum(self.test_1 * x),1],[sum(self.test_2 * x),2],[sum(self.test_3 * x),3],[sum(self.test_4 * x),4],[sum(self.test_5 * x),5],[sum(self.test_6 * x),6],[sum(self.test_7 * x),7],[sum(self.test_8 * x),8],[sum(self.test_9 * x),9]])
        return label

    def predict(self, x):
    	return self.score(x)
 
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.k = k
        self.input_dimension = input_dimension
        
    def score(self, test_data):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        distances = [(ut.euclidean_distance(test_data, image), label)
                    for (image, label) in zip(self.desc, self.label)]
        sort_distance = sorted(distances, key=lambda x: x[0])
        labels = [label for (_,label) in sort_distance[:self.k]]
        return ut.majority(labels)
                      
    def predict(self, test_data):
        """ rend la prediction sur x (-1 ou +1)
            test_data: une description : un ndarray
        """
        return self.score(test_data)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        self.desc = desc_set
        self.label = label_set


#------------------------------------
class ClassifierPerceptronKernel(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension,learning_rate, kernel):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.kernel_dimension = int(1+2*self.dimension+(mt.factorial(self.dimension)/(mt.factorial(2)*mt.factorial(self.dimension-2))))
        self.epsilon = learning_rate
        self.kernel = kernel
        self.test_0 = [0] * self.kernel_dimension
        self.test_1 = [0] * self.kernel_dimension
        self.test_2 = [0] * self.kernel_dimension
        self.test_3 = [0] * self.kernel_dimension
        self.test_4 = [0] * self.kernel_dimension
        self.test_5 = [0] * self.kernel_dimension
        self.test_6 = [0] * self.kernel_dimension
        self.test_7 = [0] * self.kernel_dimension
        self.test_8 = [0] * self.kernel_dimension
        self.test_9 = [0] * self.kernel_dimension

    def train(self, desc_set, label_set,nb_iteration):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        for j in range(nb_iteration):
        	for i in range(len(desc_set)):
        		if 0 == label_set[i]:
        			self.test_0 = self.test_0 + (np.array(self.kernel.transform(desc_set[i],int(self.kernel_dimension))) * self.epsilon)

        		if 1 == label_set[i]:
        			self.test_1 = self.test_1 + (np.array(self.kernel.transform(desc_set[i],int(self.kernel_dimension))) * self.epsilon)

        		if 2 == label_set[i]:
        			self.test_2 = self.test_2 + (np.array(self.kernel.transform(desc_set[i],int(self.kernel_dimension))) * self.epsilon)

        		if 3 == label_set[i]:
        			self.test_3 = self.test_3 + (np.array(self.kernel.transform(desc_set[i],int(self.kernel_dimension))) * self.epsilon)

        		if 4 == label_set[i]:
        			self.test_4 = self.test_4 + (np.array(self.kernel.transform(desc_set[i],int(self.kernel_dimension))) * self.epsilon)

        		if 5 == label_set[i]:
        			self.test_5 = self.test_5 + (np.array(self.kernel.transform(desc_set[i],int(self.kernel_dimension))) * self.epsilon)

        		if 6 == label_set[i]:
        			self.test_6 = self.test_6 + (np.array(self.kernel.transform(desc_set[i],int(self.kernel_dimension))) * self.epsilon)

        		if 7 == label_set[i]:
        			self.test_7 = self.test_7 + (np.array(self.kernel.transform(desc_set[i],int(self.kernel_dimension))) * self.epsilon)

        		if 8 == label_set[i]:
        			self.test_8 = self.test_8 + (np.array(self.kernel.transform(desc_set[i],int(self.kernel_dimension))) * self.epsilon)

        		if 9 == label_set[i]:
        			self.test_9 = self.test_9 + (np.array(self.kernel.transform(desc_set[i],int(self.kernel_dimension))) * self.epsilon)


    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        maxi_sum,label = max([[sum(self.test_0 * self.kernel.transform(x,int(self.kernel_dimension))),0],[sum(self.test_1 * self.kernel.transform(x,int(self.kernel_dimension))),1],[sum(self.test_2 * self.kernel.transform(x,int(self.kernel_dimension))),2],[sum(self.test_3 * self.kernel.transform(x,int(self.kernel_dimension))),3],[sum(self.test_4 * self.kernel.transform(x,int(self.kernel_dimension))),4],[sum(self.test_5 * self.kernel.transform(x,int(self.kernel_dimension))),5],[sum(self.test_6 * self.kernel.transform(x,int(self.kernel_dimension))),6],[sum(self.test_7 * self.kernel.transform(x,int(self.kernel_dimension))),7],[sum(self.test_8 *self.kernel.transform(x,int(self.kernel_dimension))),8],[sum(self.test_9 * self.kernel.transform(x,int(self.kernel_dimension))),9]])
        return label

    def predict(self, x):
    	return self.score(x)