# model_trainer.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- Carga de Datos Optimizada ---
# Cargamos el dataset completo preprocesado desde el archivo .pkl una sola vez.
print("Cargando dataset preprocesado...")
try:
    with open('preprocessed_spam_data.pkl', 'rb') as f:
        X_full, y_full = pickle.load(f)
    print(f"Dataset cargado en memoria. {len(X_full)} correos listos.")
except FileNotFoundError:
    print("Error: El archivo 'preprocessed_spam_data.pkl' no se encontró.")
    print("Asegúrate de ejecutar 'preprocess_dataset.py' primero y tener el archivo en el directorio.")
    X_full, y_full = [], [] # Evita que la app crashee si el archivo no existe

def train_and_evaluate(num_training_samples, num_test_samples=2000, status_callback=None):
    """
    Entrena y evalúa un pipeline de Regresión Logística usando datos precargados.
    
    Esta función es ahora mucho más eficiente, ya que la parte más lenta (lectura y limpieza de archivos)
    ya se hizo en el script de preprocesamiento.
    """
    if status_callback:
        status_callback("Preparando conjuntos de entrenamiento y prueba...", 0.1)

    # 1. Dividir los datos precargados en memoria
    X_train = X_full[:num_training_samples]
    y_train = y_full[:num_training_samples]
    # Usamos un conjunto de prueba fijo para que las comparaciones sean justas
    X_test = X_full[num_training_samples : num_training_samples + num_test_samples]
    y_test = y_full[num_training_samples : num_training_samples + num_test_samples]

    # 2. Definir el Pipeline de Machine Learning (¡Exacto al del notebook!)
    #    - TfidfVectorizer: Convierte texto en vectores numéricos basados en frecuencia e importancia.
    #    - LogisticRegression: El clasificador con los parámetros optimizados.
    text_clf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=150, decode_error='ignore')),
        ('clf', LogisticRegression(
            random_state=42, 
            solver='liblinear', 
            max_iter=100, # Aumentamos por si acaso, aunque liblinear converge rápido
            C=0.066
        )),
    ])

    # 3. Entrenar el Pipeline completo
    if status_callback:
        status_callback("Entrenando el Pipeline (TF-IDF + Regresión Logística)...", 0.5)
    text_clf_pipeline.fit(X_train, y_train)

    # 4. Realizar predicciones y calcular métricas
    if status_callback:
        status_callback("Realizando predicciones en el conjunto de prueba...", 0.8)
    y_pred = text_clf_pipeline.predict(X_test)
    y_pred_proba = text_clf_pipeline.predict_proba(X_test)[:, 1] # Probabilidades para la clase 'spam'
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    conf_matrix = confusion_matrix(y_test, y_pred, labels=text_clf_pipeline.classes_)
     
    if status_callback:
        status_callback("Evaluación completa.", 1)

    # Devolvemos el pipeline entrenado y todos los resultados necesarios para las visualizaciones
    return {
        "classifier": text_clf_pipeline, # Devolvemos el pipeline completo
        "X_test": X_test,               # El texto original para que `from_estimator` funcione
        "y_test": y_test,
        "y_pred_proba": y_pred_proba,
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "classes": text_clf_pipeline.classes_
    }