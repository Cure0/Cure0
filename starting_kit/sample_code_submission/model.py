'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import IncrementalPCA

class model (BaseEstimator):
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
		'''
		les differentes fonctions qui peuvent factoriser la matrice sont :
			-TruncatedSVD
			-PCA
			-KernelPCA
			-SparcePCA
			-IncrementalPCA
		'''
		self.svd = TruncatedSVD(n_components = 10)
		self.kernel = KernelPCA(n_components = 10)
        self.pca = PCA(n_components = 10)
		self.sparse = SparcePCA(n_components = 10)
		self.incremental = IncrementalPCA(n_components = 10)
        self.classifier= SVC(C=1.)

	
    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
		
        
		Cette fonction devrait entraîner les paramètres du modèle.
        Ici on ne fait rien dans cet exemple ...
        Args:
            X: matrice de données d'apprentissage de dim num_train_samples * num_feat.
            y: matrice d'étiquette d'apprentissage de dim num_train_samples * num_labels.
        Les deux entrées sont des tableaux numpy.
        Pour la classification, les étiquettes peuvent être les chiffres 0, 1, ... c-1 pour c classe
        ou vecteur codé à chaud de zéros, avec un 1 à la kième position pour la classe k.
        Le format AutoML prend en charge le codage à chaud, qui fonctionne également pour les problèmes multi-libellés.
        Utilisez data_converter.convert_to_num () pour convertir le fichier en format de numéro de catégorie.
        Pour la régression, les étiquettes sont des valeurs continues.
        
        '''
        #self.svd.fit(X)
		
		'''
		Nous avons comparer dans le notebook les différentes fonctions et le résultat obtenue pour le RMSE de chacun et choisis la fonction 
		Avec le RMSE le plus faible, il s'agissait de SparcePCA
		'''
		
        X = self.sparse.fit_transform(X)
        self.classifier.fit(X,y)
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
		
		
		Cette fonction devrait fournir des prédictions d'étiquettes sur des données (de test).
        Ici nous venons de retourner des zéros ...
        Assurez-vous que les valeurs prédites sont au bon format pour le scoring
        métrique. Par exemple, les problèmes de classification binaire s’attendent souvent à des prédictions
        sous la forme d'une valeur discriminante (si l'aire sous la courbe ROC est la métrique)
        ce sont plutôt les prédictions de la classe qui s’appliquent. Pour multi-classes ou multi-étiquettes
        problèmes, des probabilités de classe sont souvent attendues si la métrique est à entropie croisée.
        Scikit-learn a également une fonction Predict-Proba, nous n'en avons pas besoin.
        La fonction prédire éventuellement peut renvoyer des probabilités.
        '''
    #    print("Matrix Factorization of test set by SVD")
    #    svd = TruncatedSVD(n_components = 100)

    #    A = svd.fit_transform(X)
    #    T = svd.components_
        
#        print("Shape of A :", A.shape)
#        print("Shape of T :", T.shape)
        X = self.pca.fit_transform(X)
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = self.classifier.predict(X)
        # If you uncomment the next line, you get pretty good results for the Iris data :-)
  #      y = np.round(X[:,3])
        return y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
