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
        self.w = np.random.uniform(-1,1,input_dimension)
        self.input_dimension = input_dimension
        self.epsilon = learning_rate
        
    def train(self, desc_set, label_set,nb_iteration):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """    
        for i in range(nb_iteration):
            tab = list(range(len(desc_set)))
            random.shuffle(tab)
            for i in tab:
                prediction = self.predict(desc_set[i])
                if prediction != label_set[i]:
                    self.w = self.w + (self.epsilon * label_set[i]) * desc_set[i]

    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return sum(self.w * x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0:
            return -1
        return 1
 
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


# ---------------------------


class ClassifierPerceptronKernel(Classifier):
    def __init__(self,learning_rate,kernel,input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension(int) : dimension de la description des exemples
                - kernel : choix de la transformation
                - learning_rate :
            Hypothèse : kernel_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = learning_rate
        self.kernel = kernel
        # kernel_dimension = 1+2n+( n! / ( 2! * (n-2)! ) )
        self.kernel_dimension = 1+2*self.dimension+(mt.factorial(self.dimension)/(mt.factorial(2)*mt.factorial(self.dimension-2)))
        self.w = np.random.uniform(-1,1,int(self.kernel_dimension))
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return sum(self.w * self.kernel.transform(x,int(self.kernel_dimension)))
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0:
            return -1
        return 1

    def train(self, desc_set, label_set, nb_iteration):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """           
        for i in range(nb_iteration):
        	tab = list(range(len(desc_set)))
        	random.shuffle(tab)
        	for i in tab:
        		prediction = self.predict(desc_set[i])
        		if prediction != label_set[i]:
        			self.w = self.w + (self.epsilon * label_set[i]) * np.array(self.kernel.transform(desc_set[i],int(self.kernel_dimension)))

