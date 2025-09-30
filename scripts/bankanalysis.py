import clasificadores

#---------------------------
# Exploración de los datos
#---------------------------
print("\n---------Exploración de los datos------\n")
#Leer los datos, en este caso se lee de la carpeta si se va a leer directamente del link como dijo Antonio cambiar la siguiente linea
df = pd.read_csv('bank-additional-full.csv', sep=";",encoding='latin-1',na_values=['NA','NA ', 'NaN', 'null', ''])
print(df.head())
# Dimension de la base
print(f"La base de datos esta conformada por {df.shape[1]} columnas y {df.shape[0]} filas")
#Visualizacion de los tipos de datos
print("\nVisualizacion de los datos\n",df.dtypes)
# Ver si hay datos faltantes
print(df.isna().sum())
# Las columnas de tipo objecto son de categorias
# Ver el numero y categorias de cada columna
print("\n---------Columnas categoricas------\n")
cols_categoricas = ['job', 'marital', 'education','default','housing','loan','contact','month','day_of_week','poutcome']
for col in cols_categoricas:
    categorias = df[col].unique()        # Obtiene categorías únicas de la columna
    num_categorias = len(categorias)    # Cuenta cuántas categorías únicas hay
    print(f"\nCategorías en '{col}' ({num_categorias}):")
    print(categorias)

# Numero y porcentaje de unknowns
print("\n---------unknowns------\n")
for colum in cols_categoricas:
    total = df.shape[0]
    tam = len(df[df[colum] != "unknown"])
    missing = total - tam
    porcentaje = (missing / total) * 100
    print(f"La columna {colum} tiene {missing} unknown ({porcentaje:.2f}%)")

# Imputar con moda
moda_job = df[df['job'] != 'unknown']['job'].mode()[0]
df['job'] = df['job'].replace('unknown', moda_job)
moda_marital = df[df['marital'] != 'unknown']['marital'].mode()[0]
df['marital'] = df['marital'].replace('unknown', moda_marital)

#-----------------------------------------------------
# Comenzar a analizar y separar las variables catoricas
#----------------------------------------------------
# separar las variables de entrada X y de salida y
X = df.drop("y", axis=1)
y = df["y"]
print(f"Tamaño de  X: {X.shape}, Tamaño de y: {y.shape}")
# y es una variable binaria cambiar "yes=1" y "no=0"
y = y.map({"yes": 1, "no": 0})
# Por otra parte como X tiene variables categoricas vamos a tranformas a este tipo y volvera del tipo dummy
X_cat = pd.get_dummies(X, drop_first=True)
print(f"Shape X: {X_cat.shape}")
# Separar train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_cat, y, test_size=0.3, stratify=y, random_state=42)
print("Tamaño entrenamiento:", X_train.shape)
print("Tamaño prueba:", X_test.shape)
print(f"Las personas que al final si hicieron el deposito a largo plazo fue {sum(y)}")
# Convertir a arrays de numpy para que k-NN lo soporte
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

## Trabajo con datos desbalanceados y sin limpiar

k = 5
wei = None

# Realizar pruebas
y_lda=LDA(X_train, X_test, y_train, y_test)
y_qda=QDA(X_train, X_test, y_train, y_test)
# Al correrlo marca una advertencia esto puede ocurrir por colinealidad
y_qda=QDA(X_train, X_test, y_train, y_test,0.1) # Caso regularizado
y_nb=naiveBayes(X_train, X_test, y_train, y_test)
y_knn=k_NN(X_train, X_test, y_train, y_test,k)
y_lr=logistica(X_train, X_test, y_train, y_test, pesos=wei)
#Mas valores de k
n=[1,3,5,7,11]
for i in n:
    y_knn=k_NN(X_train, X_test, y_train, y_test,i)
#=====================================
# Comparación de todos los modelos
#====================================
# Diccionario para guardar los resultados
results = {
    "Naive Bayes": obtener_metricas(y_test, y_nb),
    "LDA": obtener_metricas(y_test, y_lda),
    "QDA": obtener_metricas(y_test, y_qda),
    f"k-NN (k={k})": obtener_metricas(y_test, y_knn),
    f"Regresión Logística (pesos={wei})": obtener_metricas(y_test, y_lr)
}

# Convertir a DataFrame para visualizar
df_results = pd.DataFrame(results).T
print("\n=== Comparación Final de Modelos ===")
print(df_results.round(3))
#=====================================
# Validación cruzada comparativa
#====================================
modelos = {
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(reg_param=0.2), # Regular en caso de colinealidad
    f"k-NN (k={k})": KNeighborsClassifier(n_neighbors=k),
    f"Regresión Logística (pesos={wei})": LogisticRegression(class_weight=wei, max_iter=1000)
}

# Evaluación con accuracy
X_np = np.array(X_cat, dtype=np.float64)
y_np = np.array(y)
df_cv_acc = validar_modelos(modelos, X_np, y_np)