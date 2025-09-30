# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import clasificadores as clf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

def plot_decision_boundary(model, X, y, title,tamano,k=0):
    # Crear nueva figura cada vez
    plt.figure(figsize=(6, 5))
    
    # Crear grid en el espacio 2D
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predicciones sobre el grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Graficar fronteras y puntos
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k", s=40)
    if k == 0:
        plt.title(f"{title}\n (n={tamano})", fontsize=12, pad=10)
    else:
        plt.title(f"{title}\n(n={tamano}, k={k})", fontsize=12, pad=10)
    
    plt.show()
    
def generar_datos(mu0, mu1, Sigma0, Sigma1, n):
    """
    Genera los datos con densidad multinormal

    Parameters
    ----------
    mu0 : Vector de medias para la clase 01
    mu1 : Vector de medias para la clase 1
    Sigma0 : Matriz de covarianza para la clase 0
    Sigma1 : Matriz de covarianza para la clase 1
    n : tamaño clase (balanceado)

    Returns
    -------
    X :
    y :

    """
    X0 = np.random.multivariate_normal(mu0, Sigma0, n)
    X1 = np.random.multivariate_normal(mu1, Sigma1, n)
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

def clasificador_bayes(X, mu0, mu1, Sigma0, Sigma1, pi0=0.5, pi1=0.5):
    dens_0 = multivariate_normal.pdf(X, mean=mu0, cov=Sigma0)
    dens_1 = multivariate_normal.pdf(X, mean=mu1, cov=Sigma1)
    posterior_0 = pi0 * dens_0
    posterior_1 = pi1 * dens_1
    return (posterior_1 > posterior_0).astype(int)


def metricas_nk (mu0,mu1,Sigma0,Sigma1,n_list,k_list,pi0=0.5,pi1=0.5,reg_param=0,matriz=False,imprimir=False,frontera=True):
    """
    imprime las metricas con diferentes tamaños y vecinos

    Parameters
    ----------
    mu0 : Vector de medias para la clase 01
    mu1 : Vector de medias para la clase 1
    Sigma0 : Matriz de covarianza para la clase 0
    Sigma1 : Matriz de covarianza para la clase 1
    n_list : Lista de tamaños 
    k_list : Lista de vecinos
    pi0 : valor apriori 0
    pi1 : valor apriori 1 default 1
    reg_param : parametro de regularizacion para el metodo QDA default is 0.
    matriz : booelano por defecto es  False para no graficar  la matriz de confusion
    imprimir : booelano por defecto es  False para no imrpimir las metricas
    frontera: si queremos graficar las fronteras de clasificacion
    

    Returns
    -------
    None.

    """
    for i in n_list:
        print(f"\n-------- Tamaño de muestra  {i} --------\n")
        X, y = generar_datos(mu0, mu1, Sigma0, Sigma1, i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        # Clasificadores
        #LDA
        y_lda = clf.LDA(X_train, X_test, y_train, y_test,matriz,imprimir)
        #QDA
        y_qda = clf.QDA(X_train, X_test, y_train, y_test,reg_param, matriz,imprimir)
        #Naive Bayes
        y_nb  = clf.naiveBayes(X_train, X_test, y_train, y_test,matriz,imprimir)
        y_bayes = clasificador_bayes(X_test, mu0, mu1, Sigma0, Sigma1, pi0, pi1)
        # Imprimir las fronteras de clasificacion
        if frontera:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train, y_train)
            plot_decision_boundary(lda, X_train, y_train, "Frontera de decisión - LDA",i)
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(X_train, y_train)
            plot_decision_boundary(qda, X_train, y_train, "Frontera de decisión - QDA",i)
            nb = GaussianNB()
            nb.fit(X_train, y_train)
            plot_decision_boundary(nb, X_train, y_train, "Frontera de decisión - Naive Bayes",i)
            
        
        resultados = {
            "Bayes óptimo": clf.obtener_metricas(y_test, y_bayes),
            "Naive Bayes": clf.obtener_metricas(y_test, y_nb),
            "LDA": clf.obtener_metricas(y_test, y_lda),
            "QDA": clf.obtener_metricas(y_test, y_qda)
        }
        
        
        # k-NN
        for k in k_list:
            y_knn = clf.k_NN(X_train, X_test, y_train, y_test, k,matriz,imprimir)
            resultados[f"k-NN k={k}"] = clf.obtener_metricas(y_test, y_knn)
            if frontera:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                plot_decision_boundary(knn, X_train, y_train, "Frontera de decisión - k-NN ",i,k)
            
        df_res = pd.DataFrame(resultados).T.round(3)
        print("=== Comparación de métricas ===")
        print(df_res)
        
