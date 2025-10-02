# Paqueterías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import clasificadores as cla


def elegir_k(X, y, kmax=100, scoring="f1", cv=5, plot_title="Curva de error de kNN"):
    """
    Selecciona k maximizando la métrica 'scoring' (por defecto F1).
    - Usa StratifiedKFold para evitar folds sin la clase positiva.
    - Ajusta el k máximo para que no exceda el tamaño del fold de entrenamiento.
    - Grafica la curva métrica vs k (impares).
    """
    # Asegurar formato compatible
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X_arr = X.values
    else:
        X_arr = X
    y_arr = pd.Series(y).values  # garantiza vector 1D y binario si ya lo mapeaste a {0,1}

    # CV estratificada (o respeta si ya pasaste un objeto CV)
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if isinstance(cv, int) else cv

    # Calcula splits una vez (también nos da el tamaño mínimo de train por fold)
    splits = list(cv_obj.split(X_arr, y_arr))
    n_train_min = min(len(tr) for tr, _ in splits)

    # Cota superior segura para k (al menos 1, y menor que el tamaño del set de entrenamiento del fold más chico)
    k_upper = max(1, min(kmax, n_train_min - 1))

    # Probamos k impares (si quedara vacío por alguna razón, caemos a [1])
    k_range = [k for k in range(1, k_upper + 1) if k % 2 == 1] or [1]

    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        # Reutilizamos 'splits' para que la estratificación y los tamaños sean consistentes
        cv_scores = cross_val_score(
            knn, X_arr, y_arr, cv=splits, scoring=scoring, n_jobs=-1, error_score="raise"
        )
        scores.append(cv_scores.mean())

    best_idx = int(np.argmax(scores))
    best_k = k_range[best_idx]

    # Curva
    plt.figure()
    plt.plot(k_range, scores)
    plt.xlabel("k")
    plt.ylabel(f"{scoring.upper()} promedio (CV={cv_obj.get_n_splits() if hasattr(cv_obj,'get_n_splits') else 'custom'})")
    plt.title(f"{plot_title} (mejor k={best_k})")
    plt.tight_layout()
    plt.show()

    return best_k


# ==========================
# 1) Lectura + imputación
# ==========================
print("\n---------Exploración de los datos------\n")
df = pd.read_csv("bank-additional-full.csv", sep=";", encoding="latin-1",
                 na_values=["NA", "NA ", "NaN", "null", ""])

print(df.head())
print(f"La base de datos tiene {df.shape[1]} columnas y {df.shape[0]} filas")
print("\nTipos:\n", df.dtypes)
print("\nNulos:\n", df.isna().sum())

cols_categoricas = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
print("\n---------unknowns------\n")
for col in cols_categoricas:
    total = len(df)
    missing = (df[col] == "unknown").sum()
    print(f"{col:12s}: {missing} unknown ({missing/total*100:.2f}%)")

# Imputación simple por moda (puedes extenderla al resto si quieres)
for col in ['job', 'marital']:
    moda = df.loc[df[col] != 'unknown', col].mode()[0]
    df[col] = df[col].replace('unknown', moda)

# ==========================
# 2) Dummies + split
# ==========================
X = df.drop(columns=["y"])
y = df["y"].map({"yes": 1, "no": 0})

X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
print("Tamaño entrenamiento:", X_train.shape, " | prueba:", X_test.shape)
print(f"Positivos totales (yes): {y.sum()}")

# ==========================
# (A) Cinco clasificadores con imputación
# ==========================
best_k = elegir_k(X_train, y_train, kmax=100, scoring="f1", cv=5,
                  plot_title="Curva de error de kNN (datos imputados)")

best_k = 41

y_lda, auc_lda = cla.LDA(X_train, X_test, y_train, y_test)
y_qda, auc_qda = cla.QDA(X_train, X_test, y_train, y_test, reg_param=0.1)
y_nb,  auc_nb  = cla.naiveBayes(X_train, X_test, y_train, y_test)
y_knn, auc_knn = cla.k_NN(X_train, X_test, y_train, y_test, n_neighbors=best_k)
y_lr,  auc_lr  = cla.logistica(X_train, X_test, y_train, y_test, pesos=None)

results_A = {
    "Naive Bayes":             cla.obtener_metricas(y_test, y_nb,  auc_nb),
    "LDA":                     cla.obtener_metricas(y_test, y_lda, auc_lda),
    "QDA (reg=0.1)":           cla.obtener_metricas(y_test, y_qda, auc_qda),
    f"k-NN (k={best_k})":      cla.obtener_metricas(y_test, y_knn, auc_knn),
    "Regresión Logística":     cla.obtener_metricas(y_test, y_lr,  auc_lr),
}
df_A = pd.DataFrame(results_A).T
print("\n=== (A) Comparación cinco clasificadores (imputados) ===")
print(df_A.round(3))

# Validación cruzada comparativa (puedes cambiar scoring a 'f1')
modelos_A = {
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA (reg=0.1)": QuadraticDiscriminantAnalysis(reg_param=0.1),
    f"k-NN (k={best_k})": KNeighborsClassifier(n_neighbors=best_k),
    "Regresión Logística": LogisticRegression(solver="liblinear", max_iter=1000)
}
_ = cla.validar_modelos(modelos_A, X.values, y.values, scoring="accuracy", n_splits=5)


# ==========================
# (B) Logística pesada con imputación
# ==========================
y_lr_bal, auc_lr_bal = cla.logistica(X_train, X_test, y_train, y_test, pesos="balanced")
results_B = {
    "Regresión Logística (balanced)": cla.obtener_metricas(y_test, y_lr_bal, auc_lr_bal)
}
df_B = pd.DataFrame(results_B).T
print("\n=== (B) Logística pesada (imputados) ===")
print(df_B.round(3))

modelos_B = {"Regresión Logística (balanced)": LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000)}
_ = cla.validar_modelos(modelos_B, X.values, y.values, scoring="accuracy", n_splits=5)


# ==========================
# (C) Submuestreo 2:1 + cinco clasificadores + logística pesada
# ==========================
random_seed = 810
target_ratio = 2  # mayoritaria : minoritaria = 2 : 1

# Mantén pandas para indexado
y_train_ser = y_train.copy()
X_train_df  = X_train.copy()

print("\nConteos originales (train):")
print(y_train_ser.value_counts())

mask_min = (y_train_ser == 1)
X_min, y_min = X_train_df[mask_min], y_train_ser[mask_min]
X_maj, y_maj = X_train_df[~mask_min], y_train_ser[~mask_min]

n_min, n_maj = len(y_min), len(y_maj)
n_maj_target = min(int(target_ratio * n_min), n_maj)

print(f"Minoritaria (yes): {n_min} | Mayoritaria original (no): {n_maj} | Objetivo mayoritaria: {n_maj_target}")

X_maj_down = X_maj.sample(n=n_maj_target, random_state=random_seed)
y_maj_down = y_maj.loc[X_maj_down.index]

X_train_res = pd.concat([X_min, X_maj_down], axis=0)
y_train_res = pd.concat([y_min, y_maj_down], axis=0)

X_train_res = X_train_res.sample(frac=1.0, random_state=random_seed)  # shuffle
y_train_res = y_train_res.loc[X_train_res.index]

print("\nConteos tras submuestreo:")
print(y_train_res.value_counts())

best_k_res = elegir_k(X_train_res, y_train_res, kmax=100, scoring="f1", cv=5,
                      plot_title="Curva de error de kNN (submuestreo 2:1)")

y_nb_c,  auc_nb_c  = cla.naiveBayes(X_train_res, X_test, y_train_res, y_test)
y_lda_c, auc_lda_c = cla.LDA(X_train_res, X_test, y_train_res, y_test)
y_qda_c, auc_qda_c = cla.QDA(X_train_res, X_test, y_train_res, y_test, reg_param=0.1)
y_knn_c, auc_knn_c = cla.k_NN(X_train_res, X_test, y_train_res, y_test, n_neighbors=best_k_res)
y_lr_c,  auc_lr_c  = cla.logistica(X_train_res, X_test, y_train_res, y_test, pesos=None)
y_lr_bal_c, auc_lr_bal_c = cla.logistica(X_train_res, X_test, y_train_res, y_test, pesos="balanced")

results_C = {
    "Naive Bayes (2:1)":            cla.obtener_metricas(y_test, y_nb_c,  auc_nb_c),
    "LDA (2:1)":                    cla.obtener_metricas(y_test, y_lda_c, auc_lda_c),
    "QDA reg=0.1 (2:1)":            cla.obtener_metricas(y_test, y_qda_c, auc_qda_c),
    f"k-NN k={best_k_res} (2:1)":   cla.obtener_metricas(y_test, y_knn_c, auc_knn_c),
    "Logística (2:1)":              cla.obtener_metricas(y_test, y_lr_c,  auc_lr_c),
    "Logística balanced (2:1)":     cla.obtener_metricas(y_test, y_lr_bal_c, auc_lr_bal_c),
}

df_C = pd.DataFrame(results_C).T
print("\n=== (C) Comparación con submuestreo 2:1 ===")
print(df_C.round(3))
