# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
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
    
def generar_datos(mu0, mu1, Sigma0, Sigma1, n, pi0=0.5, pi1=0.5):
    """
    Genera los datos con densidad multinormal

    Parameters
    ----------
    mu0 : Vector de medias para la clase 0
    mu1 : Vector de medias para la clase 1
    Sigma0 : Matriz de covarianza para la clase 0
    Sigma1 : Matriz de covarianza para la clase 1
    n : tamaño de referencia (será ajustado por priors)
    pi0, pi1 : probabilidades a priori

    Returns
    -------
    X, y : datos y etiquetas
    """
    # Calcular tamaños reales basados en priors
    n_total = 2 * n  # Mantenemos n como referencia por clase balanceada
    n0 = int(n_total * pi0)
    n1 = n_total - n0  # Para asegurar n0 + n1 = n_total
    
    X0 = np.random.multivariate_normal(mu0, Sigma0, n0)
    X1 = np.random.multivariate_normal(mu1, Sigma1, n1)
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n0), np.ones(n1)))
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
        
def riesgo_verdadero_bayes(mu0, mu1, Sigma0, Sigma1,  pi0=0.5, pi1=0.5,n=50000, n_remuestreos=100):
    """
    Calcula el riesgo verdadero del clasificador de Bayes con múltiples remuestreos.
    
    Parameters:
    -----------
    mu0, mu1 : Medias de las clases 0 y 1
    Sigma0, Sigma1 : Matrices de covarianza
    n: parámetro por compatibilidad (no se usa)
    pi0, pi1 : Probabilidades a priori
    n_samples : Número de muestras por remuestreo
    n_remuestreos : Número de remuestreos para calcular promedio
    
    Return:
    --------
    riesgo_promedio : Riesgo verdadero promedio
    riesgo_std : Desviación estándar del riesgo entre remuestreos
    """
    riesgos = []
    
    for _ in range(n_remuestreos):
        
        X0_test = np.random.multivariate_normal(mu0, Sigma0, n)
        X1_test = np.random.multivariate_normal(mu1, Sigma1, n)
        
        # Clasificar usando el clasificador de Bayes óptimo
        y_pred0 = clasificador_bayes(X0_test, mu0, mu1, Sigma0, Sigma1, pi0, pi1)
        y_pred1 = clasificador_bayes(X1_test, mu0, mu1, Sigma0, Sigma1, pi0, pi1)
        
        # Calcular error para este remuestreo
        error0 = np.mean(y_pred0 != 0)  # Clase 0 mal clasificada
        error1 = np.mean(y_pred1 != 1)  # Clase 1 mal clasificada
        
        riesgo = pi0 * error0 + pi1 * error1
        riesgos.append(riesgo)
    
    # Calcular promedio y desviación estándar
    riesgo_promedio = np.mean(riesgos)
    riesgo_std = np.std(riesgos)
    
    return riesgo_promedio, riesgo_std


def riesgo_vs_n(mu0, mu1, Sigma0, Sigma1, pi0=0.5, pi1=0.5,
                n_values=[50], k_values=[5], n_splits=5, imprimir=False):
    """
    Experimento que calcula el riesgo para diferentes modelos, tamaños de muestra y valores de k.
    """
    resultados = []
    
    #  Riesgo de Bayes CONSTANTE (aproximado)
    riesgo_bayes, riesgo_bayes_std = riesgo_verdadero_bayes(
        mu0, mu1, Sigma0, Sigma1, pi0, pi1)
    
    for n in n_values:
        #añadir el riesgo bayes en cada tamaño
        resultados.append({
            'n': n,
            'modelo': 'Bayes Óptimo',
            'riesgo_medio': riesgo_bayes,
            'riesgo_std': riesgo_bayes_std
        })
        
        # Generar datos de entrenamiento (con n actual)
        X, y = generar_datos(mu0, mu1, Sigma0, Sigma1, n)
        
        # Definir modelos
        modelos_base = {
            'Naive Bayes': GaussianNB(),
            'LDA': LinearDiscriminantAnalysis(),
            'QDA': QuadraticDiscriminantAnalysis()
        }
        
        # Agregar modelos k-NN para cada k
        modelos_knn = {f'k-NN (k={k})': KNeighborsClassifier(n_neighbors=k) for k in k_values}
        
        # Combinar modelos 
        modelos = {**modelos_base, **modelos_knn}
        
        # Calcular riesgo para todos los modelos
        resultados_acc = clf.validar_modelos(modelos, X, y, scoring="accuracy", n_splits=n_splits, imprimir=imprimir)
        
        # Convertir a riesgo
        for modelo_nombre in modelos.keys():
            riesgo_media = 1 - resultados_acc.loc[modelo_nombre, "Media"]
            riesgo_std = resultados_acc.loc[modelo_nombre, "Desviación"]
            
            resultados.append({
                'n': n,
                'modelo': modelo_nombre,
                'riesgo_medio': riesgo_media,
                'riesgo_std': riesgo_std
            })
    
    return pd.DataFrame(resultados)
# Con replicas

def riesgo_vs_n_con_replicas(mu0, mu1, Sigma0, Sigma1, pi0=0.5, pi1=0.5,
                            n_values=[50, 100, 200, 500], k_values=[1, 3, 5, 11, 21], 
                            R=20, n_splits=5):
    """
    Versión mejorada que incluye R réplicas independientes
    """
    resultados = []
    
    # Riesgo de Bayes CONSTANTE (aproximado)
    riesgo_bayes, riesgo_bayes_std = riesgo_verdadero_bayes(mu0, mu1, Sigma0, Sigma1, pi0, pi1)
    
    for n in n_values:
        
        for replica in range(R):
            # Generar datos para esta réplica
            X, y = generar_datos(mu0, mu1, Sigma0, Sigma1, n, pi0, pi1)
            
            # Añadir riesgo Bayes en cada réplica
            resultados.append({
                'n': n,
                'replica': replica,
                'modelo': 'Bayes Óptimo',
                'riesgo_medio': riesgo_bayes,
                'riesgo_std': riesgo_bayes_std
            })
            
            # Definir modelos
            modelos_base = {
                'Naive Bayes': GaussianNB(),
                'LDA': LinearDiscriminantAnalysis(),
                'QDA': QuadraticDiscriminantAnalysis()
            }
            
            # Agregar modelos k-NN para cada k
            modelos_knn = {f'k-NN (k={k})': KNeighborsClassifier(n_neighbors=k) for k in k_values}
            modelos = {**modelos_base, **modelos_knn}
            
            # Calcular riesgo para todos los modelos usando validación cruzada
            resultados_acc = clf.validar_modelos(modelos, X, y, scoring="accuracy", 
                                               n_splits=n_splits, imprimir=False)
            
            # Convertir a riesgo y guardar
            for modelo_nombre in modelos.keys():
                riesgo_media = 1 - resultados_acc.loc[modelo_nombre, "Media"]
                riesgo_std = resultados_acc.loc[modelo_nombre, "Desviación"]
                
                resultados.append({
                    'n': n,
                    'replica': replica,
                    'modelo': modelo_nombre,
                    'riesgo_medio': riesgo_media,
                    'riesgo_std': riesgo_std,
                    'brecha_vs_bayes': riesgo_media - riesgo_bayes
                })
    
    return pd.DataFrame(resultados)

def graficar_riesgo(resultados):
    """
    Grafica solo los errores (riesgos) para cada modelo vs tamaño de muestra.
    """
    plt.figure(figsize=(12, 8))
    
    modelos_unicos = resultados['modelo'].unique()
    
    # Graficar cada modelo
    for modelo in modelos_unicos:
        datos_modelo = resultados[resultados['modelo'] == modelo].sort_values('n')
        
        # Estilo diferente para Bayes óptimo
        if modelo == 'Bayes Óptimo':
            plt.plot(datos_modelo['n'], datos_modelo['riesgo_medio'], 
                    'k--', linewidth=3, marker='s', markersize=8, label=modelo)
        else:
            plt.plot(datos_modelo['n'], datos_modelo['riesgo_medio'], 
                    'o-', linewidth=2, markersize=6, label=modelo)
    
    plt.xlabel('Tamaño por Clase (n)')
    plt.ylabel('Error (Riesgo)')
    plt.title('Error vs Tamaño de Muestra por Modelo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    ####
    
####
#Mas graficas
####
def graficar_riesgo_completo(df_resultados, escenario=""):
    """
    Genera múltiples gráficas a partir de los resultados
    """
    # 1. Gráfica: Riesgo vs n (por método)
    plt.figure(figsize=(12, 8))
    
    # Calcular promedios por modelo y n
    df_avg = df_resultados.groupby(['n', 'modelo']).agg({
        'riesgo_medio': 'mean',
        'riesgo_std': 'mean'
    }).reset_index()
    
    modelos_unicos = df_avg['modelo'].unique()
    
    for modelo in modelos_unicos:
        datos_modelo = df_avg[df_avg['modelo'] == modelo].sort_values('n')
        
        if modelo == 'Bayes Óptimo':
            plt.plot(datos_modelo['n'], datos_modelo['riesgo_medio'], 
                    'k--', linewidth=3, marker='s', markersize=8, label=modelo)
        else:
            plt.errorbar(datos_modelo['n'], datos_modelo['riesgo_medio'], 
                        yerr=datos_modelo['riesgo_std'], capsize=4,
                        marker='o', markersize=6, label=modelo, alpha=0.8)
    
    plt.xlabel('Tamaño por Clase (n)')
    plt.ylabel('Error (Riesgo)')
    plt.title(f'Error vs Tamaño de Muestra - {escenario}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_knn_vs_k(df_resultados, escenario=""):
    """
    Gráfica específica para k-NN: Riesgo vs k para diferentes n
    """
    # Filtrar solo resultados de k-NN
    df_knn = df_resultados[df_resultados['modelo'].str.contains('k-NN')].copy()
    
    # Extraer valor de k del nombre del modelo
    df_knn['k'] = df_knn['modelo'].str.extract(r'k=(\d+)').astype(int)
    
    plt.figure(figsize=(12, 8))
    
    for n in df_knn['n'].unique():
        datos_n = df_knn[df_knn['n'] == n]
        riesgo_promedio = datos_n.groupby('k')['riesgo_medio'].mean()
        plt.plot(riesgo_promedio.index, riesgo_promedio.values, 'o-', 
                label=f'n={n}', linewidth=2, markersize=6)
    
    plt.xlabel('Número de Vecinos (k)')
    plt.ylabel('Error (Riesgo)')
    plt.title(f'k-NN: Error vs k para diferentes tamaños de muestra - {escenario}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def graficar_brechas_vs_bayes(df_resultados, escenario=""):
    """
    Gráfica de brechas L(g) - L(Bayes) vs n
    """
    # Excluir Bayes Óptimo y calcular brechas promedio
    df_sin_bayes = df_resultados[df_resultados['modelo'] != 'Bayes Óptimo']
    df_brechas = df_sin_bayes.groupby(['n', 'modelo']).agg({
        'brecha_vs_bayes': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(12, 8))
    
    modelos_unicos = df_brechas['modelo'].unique()
    for modelo in modelos_unicos:
        datos_modelo = df_brechas[df_brechas['modelo'] == modelo].sort_values('n')
        plt.plot(datos_modelo['n'], datos_modelo['brecha_vs_bayes'], 
                'o-', linewidth=2, markersize=6, label=modelo)
    
    plt.xlabel('Tamaño por Clase (n)')
    plt.ylabel('Brecha: L(g) - L(Bayes)')
    plt.title(f'Brecha vs Bayes por Tamaño de Muestra - {escenario}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()




