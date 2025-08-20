# model_trainer.py

import os
import email
import string
import nltk
from html.parser import HTMLParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- Sección 1: Utilidades de Preprocesamiento ---

# Función para asegurar que los datos de NLTK (stopwords) están descargados.
def download_nltk_data():
    """Descarga 'stopwords' de NLTK si no existen en las rutas conocidas."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Descargando el paquete 'stopwords' de NLTK...")
        nltk.download('stopwords')
        print("Descarga completa.")

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

class Parser:
    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        download_nltk_data()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.punctuation = list(string.punctuation)

    def parse(self, email_path):
        try:
            with open(email_path, errors='ignore') as e:
                msg = email.message_from_file(e)
            return None if not msg else self.get_email_content(msg)
        except IOError:
            return None

    def get_email_content(self, msg):
        subject = self.tokenize(msg['Subject']) if msg['Subject'] else []
        payload = msg.get_payload()
        body = self.get_email_body(payload, msg.get_content_type())
        return {"subject": subject, "body": body}

    def get_email_body(self, payload, content_type):
        body = []
        if isinstance(payload, str):
            if 'text/plain' in content_type:
                return self.tokenize(payload)
            elif 'text/html' in content_type:
                return self.tokenize(strip_tags(payload))
        elif isinstance(payload, list):
            for p in payload:
                if p.is_multipart():
                    body.extend(self.get_email_body(p.get_payload(), p.get_content_type()))
                else:
                    body.extend(self.get_email_body(p.get_payload(), p.get_content_type()))
        return body

    def tokenize(self, text):
        if not text: return []
        for c in self.punctuation:
            text = text.replace(c, "")
        text = text.replace("\t", " ").replace("\n", " ")
        tokens = list(filter(None, text.split(" ")))
        return [self.stemmer.stem(w.lower()) for w in tokens if w.lower() not in self.stopwords and len(w) > 1]

# --- Sección 2: Lógica de Entrenamiento y Evaluación ---

# ¡RUTA SIMPLIFICADA!
# El script startup.sh se asegura de que la carpeta ALERT esté aquí.
DATASET_PATH = "ALERT/datasets/datasets/trec07p"
INDEX_PATH = os.path.join(DATASET_PATH, "full/index")


def parse_index(path_to_index, n_elements):
    """Lee el archivo de índice y devuelve una lista de etiquetas y rutas."""
    ret_indexes = []
    if not os.path.exists(path_to_index):
        raise FileNotFoundError(f"El archivo de índice no se encontró en: {path_to_index}. "
                              "Asegúrate de que el dataset esté en el lugar correcto.")
    try:
        with open(path_to_index) as index_file:
            index_lines = index_file.readlines()
        
        n_elements = min(n_elements, len(index_lines))
        for i in range(n_elements):
            mail = index_lines[i].split(" ../")
            label = mail[0]
            path = mail[1].strip()
            # La ruta en el índice es relativa, la unimos con la ruta base
            full_path = os.path.join(BASE_PATH, path)
            ret_indexes.append({"label": label, "email_path": full_path})
    except Exception as e:
        print(f"Error al leer el índice: {e}")
        return []
    return ret_indexes

def create_prep_dataset(index_path, n_elements, status_callback=None):
    X, y = [], []
    p = Parser()
    indexes = parse_index(index_path, n_elements)
    if not indexes: return None, None

    for i, index in enumerate(indexes):
        pmail = p.parse(index["email_path"])
        if pmail:
            full_text = " ".join(pmail['subject']) + " " + " ".join(pmail['body'])
            X.append(full_text)
            y.append(index["label"])
        
        if status_callback:
            progress = (i + 1) / n_elements
            status_callback(f"Procesando correo: {i+1}/{n_elements}", progress)
    return X, y

def train_and_evaluate(num_training_samples, num_test_samples=2000, status_callback=None):
    total_samples = num_training_samples + num_test_samples
     
    if status_callback: status_callback(f"Cargando y preprocesando {total_samples} correos...", 0)
    X, y = create_prep_dataset(INDEX_PATH, total_samples, status_callback)
    if not X or not y:
        raise ValueError("No se pudieron cargar los datos. Verifica la ruta del dataset.")

    X_train, y_train = X[:num_training_samples], y[:num_training_samples]
    X_test, y_test = X[num_training_samples:], y[num_training_samples:]

    if status_callback: status_callback("Vectorizando texto...", 0.9)
    vectorizer = CountVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
     
    if status_callback: status_callback("Entrenando el modelo...", 0.95)
    clf = LogisticRegression(max_iter=2000, solver='lbfgs') 
    clf.fit(X_train_transformed, y_train)

    y_pred = clf.predict(X_test_transformed)
    y_pred_proba = clf.predict_proba(X_test_transformed)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    conf_matrix = confusion_matrix(y_test, y_pred, labels=clf.classes_)
     
    if status_callback: status_callback("Evaluación completa.", 1)

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