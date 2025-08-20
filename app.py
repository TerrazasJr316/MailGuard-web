# app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import model_trainer # Importamos nuestro script de lógica

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Detector de SPAM con Regresión Logística",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Función para descargar datos de NLTK (se cachea para no descargar cada vez) ---
#@st.cache_resource
#def setup_nltk():
#    """Descarga los datos necesarios de NLTK."""
#    model_trainer.download_nltk_data()

# Llamar a la función de configuración al inicio
#setup_nltk()

# --- Título y Descripción de la Aplicación ---
st.title("📧 Detector de SPAM con Regresión Logística")
st.markdown("""
Abstract
NSL-KDD is a data set suggested to solve some of the inherent problems of the KDD'99 data set which are mentioned in [1].
Although, this new version of the KDD data set still suffers from some of the problems discussed by McHugh [2] and may not
be a perfect representative of existing real networks, because of the lack of public data sets for network-based IDSs, we
believe it still can be applied as an effective benchmark data set to help researchers compare different intrusion detection
methods. Furthermore, the number of records in the NSL-KDD train and test sets are reasonable. This advantage makes it affordable
to run the experiments on the complete set without the need to randomly select a small portion. Consequently, evaluation results
of different research work will be consistent and comparable.

Data Files
* KDDTrain+.ARFF - The full NSL-KDD train set with binary labels in ARFF format
* KDDTrain+.TXT - The full NSL-KDD train set including attack-type labels and difficulty level in CSV format
* KDDTrain+_20Percent.ARFF - A 20% subset of the KDDTrain+.arff file
* KDDTrain+_20Percent.TXT - A 20% subset of the KDDTrain+.txt file
* KDDTest+.ARFF - The full NSL-KDD test set with binary labels in ARFF format
* KDDTest+.TXT - The full NSL-KDD test set including attack-type labels and difficulty level in CSV format
* KDDTest-21.ARFF - A subset of the KDDTest+.arff file which does not include records with difficulty level of 21 out of 21
* KDDTest-21.TXT - A subset of the KDDTest+.txt file which does not include records with difficulty level of 21 out of 21

Descargar los ficheros
https://www.unb.ca/cic/datasets/nsl.html

Referencias adicionales sobre el DataSet
References: [1] M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,”
Submitted to Second IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.

Esta aplicación web te permite entrenar un modelo de Machine Learning para detectar correos SPAM.
Puedes elegir cuántos correos usar para el entrenamiento y ver en tiempo real cómo impacta en el rendimiento del modelo.
Las métricas se calculan sobre un conjunto de prueba fijo de **2000 correos** que el modelo no ha visto durante el entrenamiento.
""")

# --- Barra Lateral de Opciones (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Parámetros del Modelo")
    
    # Slider para que el usuario elija el número de muestras de entrenamiento
    num_samples = st.slider(
        "Número de correos para entrenar:",
        min_value=1000,
        max_value=50000, # Ajustado al tamaño del dataset
        value=10000,
        step=1000,
        help="Selecciona la cantidad de datos para el entrenamiento. Un número mayor puede mejorar la precisión pero tardará más en procesar."
    )

    # Botón para iniciar el entrenamiento
    train_button = st.button("🚀 Entrenar y Evaluar Modelo", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("""

            Se propone la contrucción de aprendizaje automático capaz de predecir si un correo determinado se SPAM o no, para esto se utilizará el siguiente DataSet (Conjunto de Datos):

            2007 TREC Public Spam Corpurs
            
            The corpus trec07p contains 75,419 messages:

            25220 ham
            
            50199 spam
            
            These messages constitute all the messages delivered to a particular server between these dates:

            Sun, 8 Apr 2007 13:07:21 -0400
            
            Fri, 6 Jul 2007 07:04:53 -0400
            
            Esta app fue creada como demostración a partir de cuadernos de Jupyter. El dataset utilizado es el **TREC 2007 Spam Corpus**.
            """)
    
# --- Lógica Principal de la Aplicación ---
results_placeholder = st.empty()

if train_button:
    with results_placeholder.container():
        st.info(f"Iniciando proceso con **{num_samples}** correos de entrenamiento y 2000 de prueba.")
        st.header(f"Resultados de Entrenamiento con {num_samples} correos")
        
        progress_bar = st.progress(0, text="Iniciando proceso...")
        status_text = st.empty()

        def update_status(message, progress):
            """Función de callback para actualizar la UI desde el trainer."""
            status_text.text(message)
            progress_bar.progress(progress, text=message)

        try:
            results = model_trainer.train_and_evaluate(
                num_training_samples=num_samples,
                status_callback=update_status
            )
            
            st.subheader("📊 Métricas de Rendimiento")
            col1, col2 = st.columns(2)
            
            col1.metric(
                label="✅ Accuracy (Precisión Global)",
                value=f"{results['accuracy']:.3f}"
            )
            
            f1_color = "normal" if results['f1_score'] > 0.95 else "inverse"
            col2.metric(
                label="🎯 F1-Score (SPAM)",
                value=f"{results['f1_score']:.3f}",
                help="Métrica clave que balancea falsos positivos y negativos.",
                delta_color=f1_color
            )
            
            st.markdown("---")
            st.subheader("📈 Visualizaciones del Modelo")
            
            fig_col1, fig_col2 = st.columns(2)
            
            with fig_col1:
                st.markdown("#### Matriz de Confusión")
                fig, ax = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay(
                    confusion_matrix=results['confusion_matrix'],
                    display_labels=results['classes']
                ).plot(ax=ax, cmap='Blues', values_format='d')
                ax.set_title("Rendimiento en el set de prueba")
                st.pyplot(fig)
            
            with fig_col2:
                st.markdown("#### Curva ROC")
                fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                RocCurveDisplay.from_estimator(
                    results['classifier'],
                    results['X_test'],
                    results['y_test'],
                    ax=ax_roc,
                    pos_label='spam'
                )
                ax_roc.set_title("Capacidad de Discriminación")
                st.pyplot(fig_roc)

            st.markdown("#### Curva de Precisión-Recall (PR)")
            fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
            PrecisionRecallDisplay.from_estimator(
                results['classifier'],
                results['X_test'],
                results['y_test'],
                ax=ax_pr,
                pos_label='spam'
            )
            ax_pr.set_title("Balance entre Precisión y Recall")
            st.pyplot(fig_pr)
            
            st.success("¡El modelo fue entrenado y evaluado con éxito!")
            
        except Exception as e:
            st.error(f"Ocurrió un error durante el proceso: {e}")
            st.error("Asegúrate de que el archivo 'preprocessed_spam_data.pkl' se haya descargado correctamente.")

else:
    with results_placeholder.container():
        st.info("Configura los parámetros en la barra lateral y presiona 'Entrenar y Evaluar Modelo' para comenzar.")