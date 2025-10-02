import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def _to_numpy_c(X, dtype=float):
    # DataFrame/Series → ndarray sin copia si es posible, luego forzamos C-contiguo
    if isinstance(X, (pd.DataFrame, pd.Series)):
        arr = X.to_numpy(dtype=dtype, copy=False)
    else:
        arr = np.asarray(X, dtype=dtype)
    return np.ascontiguousarray(arr)

# ====== Utilidades ======
def _safe_auc(y_true, model, X, y_pred=None):
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
            return roc_auc_score(y_true, y_score)
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X)
            return roc_auc_score(y_true, y_score)
        # fallback (no ideal)
        if y_pred is not None:
            return roc_auc_score(y_true, y_pred)
    except Exception:
        pass
    return None

def grafica_confusion(cm, metodo, n_neighbors=None, pesos=None):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    if metodo.lower() == "k-nn":
        titulo = f'Matriz de confusión - {metodo} (k={n_neighbors})'
    elif metodo.lower() == "regresión logística":
        titulo = f'Matriz de confusión - {metodo}' + (f' (pesos={pesos})' if pesos is not None else '')
    else:
        titulo = f'Matriz de confusión - {metodo}'
    plt.title(titulo)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

def mostrar_metricas(y_true, y_pred, auc=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precisión (weighted):", precision_score(y_true, y_pred, average='weighted', zero_division=0))
    print("Sensibilidad (Recall, weighted):", recall_score(y_true, y_pred, average='weighted', zero_division=0))
    print("Especificidad:", especificidad)
    print("F1-score (weighted):", f1_score(y_true, y_pred, average='weighted', zero_division=0))
    print("AUC:" if auc is not None else "AUC: N/A (requiere probabilidades)", auc if auc is not None else "")

def obtener_metricas(y_true, y_pred, auc=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    # Mantengo AUC sólo si alguien ya lo calcula afuera con probas; aquí lo dejamos N/A.
    return {
        "Acc": accuracy_score(y_true, y_pred),
        "Precisión": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "Especificidad": especificidad,
        "F1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "AUC": auc if auc is not None else np.nan
    }


# ====== Modelos ======
def naiveBayes(X_train, X_test, y_train, y_test, matriz=True, imprimir=True):
    nb = GaussianNB().fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    auc = _safe_auc(y_test, nb, X_test, y_pred)

    if imprimir:
        print("\n--------  Naive Bayes --------\n")
        print("Matriz de confusión:\n", cm)
        mostrar_metricas(y_test, y_pred, auc)
    if matriz:
        grafica_confusion(cm, 'Naive Bayes')
    return y_pred, auc

def LDA(X_train, X_test, y_train, y_test, matriz=True, imprimir=True):
    lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    auc = _safe_auc(y_test, lda, X_test, y_pred)

    if imprimir:
        print("\n--------  LDA --------\n")
        print("Matriz de confusión:\n", cm)
        mostrar_metricas(y_test, y_pred, auc)
    if matriz:
        grafica_confusion(cm, 'LDA')
    return y_pred, auc

def QDA(X_train, X_test, y_train, y_test, reg_param=0.0, matriz=True, imprimir=True):
    qda = QuadraticDiscriminantAnalysis(reg_param=reg_param).fit(X_train, y_train)
    y_pred = qda.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    auc = _safe_auc(y_test, qda, X_test, y_pred)

    if imprimir:
        print("\n--------  QDA --------\n")
        print("Matriz de confusión:\n", cm)
        mostrar_metricas(y_test, y_pred, auc)
    if matriz:
        grafica_confusion(cm, 'QDA')
    return y_pred, auc

def k_NN(X_train, X_test, y_train, y_test, n_neighbors, matriz=True, imprimir=True):
    # ← NEW: garantizar formato compatible con backend de sklearn
    Xtr = _to_numpy_c(X_train)
    Xte = _to_numpy_c(X_test)
    ytr = np.asarray(y_train).ravel()

    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(Xtr, ytr)
    y_pred = knn.predict(Xte)

    cm = confusion_matrix(y_test, y_pred)
    auc = _safe_auc(y_test, knn, Xte, y_pred)

    if imprimir:
        print(f"\n--------  k-NN (k={n_neighbors}) --------\n")
        print("Matriz de confusión:\n", cm)
        mostrar_metricas(y_test, y_pred, auc)
    if matriz:
        grafica_confusion(cm, 'k-NN', n_neighbors=n_neighbors)
    return y_pred, auc

def logistica(X_train, X_test, y_train, y_test, pesos=None, matriz=True, imprimir=True):
    logreg = LogisticRegression(solver="liblinear", class_weight=pesos, max_iter=2000).fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    auc = _safe_auc(y_test, logreg, X_test, y_pred)

    if imprimir:
        print(f"\n--------  Regresión Logística (pesos={pesos}) --------\n")
        print("Matriz de confusión:\n", cm)
        mostrar_metricas(y_test, y_pred, auc)
    if matriz:
        grafica_confusion(cm, 'Regresión Logística', pesos=pesos)
    return y_pred, auc


# ====== Validación cruzada multip modelos ======
def validar_modelos(modelos, X, y, scoring="accuracy", n_splits=5, imprimir=True):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    resultados = {}
    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring)
        resultados[nombre] = {"Media": scores.mean(), "Desviación": scores.std()}

    if imprimir:
        print(f"\n=== Validación Cruzada ({n_splits}-fold, {scoring}) ===")
        for nombre in modelos.keys():
            m = resultados[nombre]["Media"]
            s = resultados[nombre]["Desviación"]
            print(f"{nombre:30s}: {m:.3f} ± {s:.3f}")
    return pd.DataFrame(resultados).T
