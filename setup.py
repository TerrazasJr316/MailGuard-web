# setup.py
import streamlit as st
import requests
import zipfile
import os
import shutil

# --- Configuración ---
# IMPORTANTE: Reemplaza esta URL con el enlace de descarga directa de tu archivo ZIP.
DATASET_URL = "URL_DE_DESCARGA_DIRECTA_AQUI"

# Rutas de destino
TARGET_DIR = "ALERT/datasets"
ZIP_PATH = os.path.join(TARGET_DIR, "dataset.zip")
EXTRACT_PATH = TARGET_DIR
EXPECTED_FOLDER = os.path.join(EXTRACT_PATH, "trec07p") # La carpeta que debe existir después de descomprimir

def download_and_unzip():
    """
    Descarga y descomprime el dataset si no existe localmente.
    Esta función está diseñada para ser llamada una sola vez gracias a st.cache_resource.
    """
    # Si la carpeta final ya existe, no hacemos nada.
    if os.path.exists(EXPECTED_FOLDER):
        print("El dataset ya existe. Omitiendo descarga.")
        return

    st.info(f"Preparando el entorno por primera vez. Descargando dataset desde la nube... (esto puede tardar varios minutos)")
    
    # Asegurarse de que el directorio de destino existe
    os.makedirs(TARGET_DIR, exist_ok=True)

    try:
        # Descargar el archivo ZIP
        with requests.get(DATASET_URL, stream=True) as r:
            r.raise_for_status()
            with open(ZIP_PATH, 'wb') as f:
                # Usar shutil para manejar la descarga de manera eficiente
                shutil.copyfileobj(r.raw, f)
        
        print(f"Dataset descargado en {ZIP_PATH}")

        # Descomprimir el archivo
        st.info("Dataset descargado. Descomprimiendo archivos...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        
        print(f"Dataset descomprimido en {EXTRACT_PATH}")

        # Limpiar el archivo ZIP después de la extracción
        os.remove(ZIP_PATH)
        print(f"Archivo ZIP eliminado: {ZIP_PATH}")

        # Verificar que la carpeta esperada se haya creado
        if not os.path.exists(EXPECTED_FOLDER):
            raise FileNotFoundError(f"La descompresión falló. No se encontró la carpeta '{EXPECTED_FOLDER}'. "
                                    f"Asegúrate de que el ZIP contenga la carpeta 'trec07p' en su raíz.")
        
        st.success("¡Entorno listo! La aplicación ya está operativa.")

    except Exception as e:
        st.error(f"Ocurrió un error crítico durante la configuración: {e}")
        st.error("La aplicación no puede continuar. Por favor, verifica la URL del dataset y la estructura del archivo ZIP.")
        # Detener la ejecución de la app si la configuración falla
        st.stop()

