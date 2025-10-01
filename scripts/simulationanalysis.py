# -*- coding: utf-8 -*-

import numpy as np
import funcionesP2 as fp2

#------------
# Caso 1
#------------
#  parámetros
mu0 = np.array([0, 0])
mu1 = np.array([1, 1])
Sigma0 = np.array([[1, 0.5], [0.5, 1]])
Sigma1 = np.array([[1, -0.3], [-0.3, 1]])
n_values= [50, 100, 200, 500]
k_values=[1, 3, 5, 7]
pi0=0.5
pi1=0.5
R=20
#  Métricas para diferentes tamaños
fp2.metricas_nk(mu0, mu1, Sigma0, Sigma1,n_values, k_values, frontera=False)
#si quieres graficar las fronteras aunque si n y k tienen muchos elementos tardara
# ya que seran (len(k)+3)*len(n) graficas  se sugiere usar vectores pequeños
#fp2.metricas_nk(mu0, mu1, Sigma0, Sigma1, [100,200], [3])


#  Análisis de riesgo vs n
resultados = fp2.riesgo_vs_n(mu0, mu1, Sigma0, Sigma1,pi0,pi1,n_values, k_values)

#  Graficar resultados
fp2.graficar_riesgo(resultados)


# =============================================================================
# ESCENARIO 2: COVARIANZAS DISTINTAS (QDA ÓPTIMO)
# =============================================================================

mu0_2 = np.array([0, 0])
mu1_2 = np.array([1, 1])
Sigma0_2 = np.array([[1, 0.5], [0.5, 1]])
Sigma1_2 = np.array([[2, -0.5], [-0.5, 1]])  # Covarianzas diferentes
pi0_2, pi1_2 = 0.5, 0.5

# Métricas rápidas
fp2.metricas_nk(mu0_2, mu1_2, Sigma0_2, Sigma1_2, [100, 200], [3], frontera=False)

# Análisis con réplicas
resultados_2 = fp2.riesgo_vs_n_con_replicas(
    mu0_2, mu1_2, Sigma0_2, Sigma1_2, pi0_2, pi1_2,
    n_values, k_values, R=20)

# Gráficas
fp2.graficar_riesgo(resultados_2)
fp2.graficar_knn_vs_k(resultados_2, "Covarianzas Distintas")
fp2.graficar_brechas_vs_bayes(resultados_2, "Covarianzas Distintas")

# =============================================================================
# ESCENARIO 3: DESBALANCE DE CLASES
# =============================================================================

mu0_3 = np.array([0, 0])
mu1_3 = np.array([1.5, 1.5])  # Mayor separación para compensar desbalance
Sigma0_3 = np.array([[1, 0.3], [0.3, 1]])
Sigma1_3 = np.array([[1, 0.3], [0.3, 1]])
pi0_3, pi1_3 = 0.8, 0.2  # Desbalance marcado

# Métricas rápidas
fp2.metricas_nk(mu0_3, mu1_3, Sigma0_3, Sigma1_3, [100, 200], [3], 
                pi0=pi0_3, pi1=pi1_3, frontera=False)

# Análisis con réplicas
resultados_3 = fp2.riesgo_vs_n_con_replicas(
    mu0_3, mu1_3, Sigma0_3, Sigma1_3, pi0_3, pi1_3,
    n_values, k_values, R=R
)

# Gráficas
fp2.graficar_riesgo_completo(resultados_3, "Desbalance (π₀=0.8, π₁=0.2)")
fp2.graficar_knn_vs_k(resultados_3, "Desbalance (π₀=0.8, π₁=0.2)")
fp2.graficar_brechas_vs_bayes(resultados_3, "Desbalance (π₀=0.8, π₁=0.2)")

# =============================================================================
# ESCENARIO 4: ALTA CORRELACIÓN / MAL CONDICIONAMIENTO
# =============================================================================


mu0_4 = np.array([0, 0])
mu1_4 = np.array([0.5, 0.5])  # Separación pequeña
Sigma0_4 = np.array([[1, 0.95], [0.95, 1]])  # Matriz casi singular
Sigma1_4 = np.array([[1, 0.95], [0.95, 1]])
pi0_4, pi1_4 = 0.5, 0.5

# Métricas rápidas
fp2.metricas_nk(mu0_4, mu1_4, Sigma0_4, Sigma1_4, [100, 200], [3], frontera=False)

# Análisis con réplicas
resultados_4 = fp2.riesgo_vs_n_con_replicas(
    mu0_4, mu1_4, Sigma0_4, Sigma1_4, pi0_4, pi1_4,
    n_values, k_values, R=R
)

# Gráficas
fp2.graficar_riesgo_completo(resultados_4, "Alta Correlación (ρ=0.95)")
fp2.graficar_knn_vs_k(resultados_4, "Alta Correlación (ρ=0.95)")
fp2.graficar_brechas_vs_bayes(resultados_4, "Alta Correlación (ρ=0.95)")

 