# -*- coding: utf-8 -*-

import numpy as np
from funcionesP2 import metricas_nk

# Caso Facil
np.random.seed(72)
mu0 = np.array([0, 0])
mu1 = np.array([3, 3])
Sigma0 = np.array([[1, 0], [0, 1]])
Sigma1 = np.array([[1, 0], [0, 1]])
n=[100,1000]
k=[1,3,7]

# Generar datos
np.random.seed(72)
#-------------
# Metricas
#------------
metricas_nk(mu0, mu1, Sigma0, Sigma1, n, k,frontera=False)
#si quieres graficar las fronteras aunque si n y k tienen muchos elementos tardara 
metricas_nk(mu0, mu1, Sigma0, Sigma1, n, k)
#Distintas pi0 y pi1

