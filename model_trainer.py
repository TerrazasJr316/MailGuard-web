import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Cargamos el dataset completo preprocesado desde el archivo .pkl
print("Cargando dataset preprocesado...")
with open('preprocessed_spam_data.pkl', 'rb') as f:
    X_full, y_full = pickle.load(f)
print("Dataset cargado en memoria.")

def train_and_evaluate(num_training_samples, num_test_samples=2000, status_callback=None):
    """
    Función principal que ahora usa los datos precargados.
    Es mucho más rápida y consume menos memoria.
    """
    if status_callback:
        status_callback("Datos listos. Preparando conjuntos de entrenamiento y prueba...", 0.1)

    # 1. Dividir en conjuntos de entrenamiento y prueba (¡ya no hay que procesar!)
    X_train = X_full[:num_training_samples]
    y_train = y_full[:num_training_samples]
    X_test = X_full[num_training_samples : num_training_samples + num_test_samples]
    y_test = y_full[num_training_samples : num_training_samples + num_test_samples]

    # 2. Vectorizar el texto
    if status_callback:
        status_callback("Vectorizando texto...", 0.5)
    vectorizer = CountVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
     
    # 3. Entrenar el clasificador
    if status_callback:
        status_callback("Entrenando el modelo de Regresión Logística...", 0.7)
    clf = LogisticRegression(max_iter=2000, solver='lbfgs') 
    clf.fit(X_train_transformed, y_train)

    # 4. Realizar predicciones y calcular métricas
    y_pred = clf.predict(X_test_transformed)
    y_pred_proba = clf.predict_proba(X_test_transformed)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    conf_matrix = confusion_matrix(y_test, y_pred, labels=clf.classes_)
     
    if status_callback:
        status_callback("Evaluación completa.", 1)

    return {
        "classifier": clf,
        "vectorizer": vectorizer,
        "X_test": X_test_transformed,
        "y_test": y_test,
        "y_pred_proba": y_pred_proba,
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "classes": clf.classes_
    }