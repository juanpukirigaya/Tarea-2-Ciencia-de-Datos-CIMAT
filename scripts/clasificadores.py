import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## Para futura referencia
# La función que separa el conjunto de datos en train-test
from sklearn.model_selection import train_test_split
# Las métricas de desempeño de los modelos de clasificación
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Mas paqueterias
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ====== Funciones para clasificación =====

# Para graficar la matriz de confusion
def grafica_confusion(cm, metodo, n_neighbors=None, pesos=None):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    if metodo.lower() == "k-nn":
        if pesos is not None:
            plt.title(f'Matriz de confusión - {metodo} (k={n_neighbors}, pesos={pesos})')
        else:
            plt.title(f'Matriz de confusión - {metodo} (k={n_neighbors})')
    elif metodo.lower() == "regresión logística":
        if pesos is not None:
            plt.title(f'Matriz de confusión - {metodo} (pesos={pesos})')
        else:
            plt.title(f'Matriz de confusión - {metodo}')
    else:
        plt.title(f'Matriz de confusión - {metodo}')

    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

# Imprimir las metricas
def mostrar_metricas(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    especificidad = tn / (tn + fp)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precisión:", precision_score(y_true, y_pred, average='weighted', zero_division=0))
    print("Sensibilidad (Recall):", recall_score(y_true, y_pred, average='weighted'))
    print("Especificidad:", especificidad)
    print("F1-score:", f1_score(y_true, y_pred, average='weighted', zero_division=0))
    print("AUC:", roc_auc_score(y_true, y_pred))

#=====================================
#======  Naive Bayes =================
#=====================================
def naiveBayes(X_train, X_test, y_train, y_test,matriz=True,imprimir=True):
    """
    Función que realiza clasificación con el metodo de Naive Bayes y grafica
    la matriz de confusion y regresa la predicción
    Parameters
    ----------
    X_train : Covariables a entrenar
    X_test : Covariables a validar
    y_train : Variable de salida a entrenar
    y_test : Variable de salida a validar

    """
    
    # Entrenar
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Evaluar
    y_pred_nb = nb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_nb)
    
    if imprimir:
        print("\n--------  Naive Bayes --------\n")
        print("Matriz de confusión (Naive Bayes):\n", cm)
        #Metricas
        mostrar_metricas(y_test, y_pred_nb)
    if matriz:
        # Graficar matriz de confusion
        grafica_confusion(cm, 'Naive Bayes')
    
    return y_pred_nb

#=====================================
#=LDA (Linear Discriminant Analysis)=
#=====================================
def LDA(X_train, X_test, y_train, y_test,matriz=True,imprimir=True):
    """
    Función que realiza clasificación con el metodo Linear Discriminant Analysis
    y grafica la matriz de confusion y regresa la predicción
    Parameters
    ----------
    X_train : Covariables a entrenar
    X_test : Covariables a validar
    y_train : Variable de salida a entrenar
    y_test : Variable de salida a validar

    """
    
    # Entrenar
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    # Evaluar
    y_pred_lda = lda.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_lda)
    
    #imprimir metricas
    if imprimir:
        print("\n--------  LDA (Linear Discriminant Analysis) --------\n")
        print("Matriz de confusión (LDA):\n", cm)
        #Metricas
        mostrar_metricas(y_test, y_pred_lda)
    
    if matriz:
        # Graficar matriz de confusion
        grafica_confusion(cm, 'LDA')
    
    return y_pred_lda

#=====================================
# QDA (Quadratic Discriminant Analysis) =
#=====================================
def QDA(X_train, X_test, y_train, y_test,reg_param=0,matriz=True,imprimir=True):
    """
    Función que realiza clasificación con el metodo Quadratic Discriminant Analysis
    y grafica la matriz de confusion y regresa la predicción
    Parameters
    ----------
    X_train : Covariables a entrenar
    X_test : Covariables a validar
    y_train : Variable de salida a entrenar
    y_test : Variable de salida a validar
    reg_param: valor para regularizar las matrices de covarianza de cada clase por defecto 0
    """
    
    # Entrenar
    qda = QuadraticDiscriminantAnalysis(reg_param=reg_param)
    qda.fit(X_train, y_train)
    # Evaluar
    y_pred_qda = qda.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_qda)
    
    #imprimir metricas
    if imprimir:
        print("\n--------  QDA (Quadratic Discriminant Analysis) --------\n")
        print("Matriz de confusión (QDA):\n", cm)
        mostrar_metricas(y_test, y_pred_qda)
    #graficar matriz
    if matriz:
        grafica_confusion(cm, 'QDA')
        
    return y_pred_qda

#=====================================
# k-NN (k-Nearest Neighbors) =
#=====================================
def k_NN(X_train, X_test, y_train, y_test,n_neighbors,matriz=True,imprimir=True):
    """
    Función que realiza clasificación con el metodo k-Nearest Neighbors
    y grafica la matriz de confusion y regresa la predicción
    Parameters
    ----------
    n_neighbors: numero de vecinos
    X_train : Covariables a entrenar
    X_test : Covariables a validar
    y_train : Variable de salida a entrenar
    y_test : Variable de salida a validar

    """
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)

    # Evaluar
    y_pred_knn = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_knn)
    
    if imprimir:
        print(f"\n--------  k-NN (k-Nearest Neighbors) con {n_neighbors} vecinos) --------\n")    # Entrenar
        print("Matriz de confusión (k-NN):\n", cm)
        #Metricas
        mostrar_metricas(y_test, y_pred_knn)
    # Graficar matriz de confusion
    if matriz:
        cm = confusion_matrix(y_test, y_pred_knn)
        grafica_confusion(cm, 'k-NN',n_neighbors)
        
    return y_pred_knn

#=====================================
# Regresión Logística =
#=====================================
def logistica(X_train, X_test, y_train, y_test, pesos=None,matriz=True,imprimir=True):
    """
    Función que realiza clasificación con el método de Regresión Logística,
    grafica la matriz de confusión y regresa la predicción.

    Parameters
    ----------
    pesos : dict, 'balanced' o None
        Pesos de las clases (class_weight en sklearn).
    X_train : Covariables a entrenar
    X_test : Covariables a validar
    y_train : Variable de salida a entrenar
    y_test : Variable de salida a validar
    """
    
    # Entrenar
    logreg = LogisticRegression(solver="liblinear",class_weight=pesos, max_iter=2000)
    logreg.fit(X_train, y_train)

    # Evaluar
    y_pred_log = logreg.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_log)
    

    # Métricas
    if imprimir:
        print(f"\n--------  Regresión Logística (pesos={pesos}) --------\n")
        print("Matriz de confusión (Logística):\n", cm)
        mostrar_metricas(y_test, y_pred_log)
    if matriz:
        # Graficar matriz de confusión
        grafica_confusion(cm, 'Regresión Logística', pesos=pesos)
    
    return y_pred_log

#=====================================
# Comparación de todos los modelos
#=====================================
def obtener_metricas(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    especificidad = tn / (tn + fp)
    return {
        "Acc": accuracy_score(y_true, y_pred),
        "Precisión": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "Especificidad": especificidad,
        "F1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "AUC": roc_auc_score(y_true, y_pred)
    }

def validar_modelos(modelos, X, y, scoring="accuracy", n_splits=5):
    """
    Realiza validación cruzada con múltiples modelos y devuelve los resultados.

    Parámetros:
    -----------
    modelos : Diccionario con nombre del modelo como clave y modelo sklearn como valor.
    X : Variables independientes.
    y : Variable dependiente.
    scoring :  Métrica a evaluar por default 'accuracy'
    n_splits, :Número de particiones para K-Fold.

    Retorna:
    --------
    DataFrame con media y desviación estándar para cada modelo.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    resultados = {}

    print(f"\n=== Validación Cruzada ({n_splits}-fold, {scoring}) ===")
    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring)
        media = scores.mean()
        std = scores.std()
        resultados[nombre] = {"Media": media, "Desviación": std}
        print(f"{nombre:20s}: {media:.3f} ± {std:.3f}")

    return pd.DataFrame(resultados).T






