#!/bin/bash
set -e

# Pega aquí la NUEVA URL de tu archivo .pkl
DATASET_URL="https://github.com/TerrazasJr316/MailGuard-web/releases/download/v2.0-data/preprocessed_spam_data.pkl"

# Revisa si el archivo de datos ya existe
if [ ! -f "preprocessed_spam_data.pkl" ]; then
    echo "Dataset preprocesado no encontrado. Descargando..."
    curl -L $DATASET_URL -o preprocessed_spam_data.pkl
    echo "Descarga completa."
else
    echo "Dataset preprocesado ya existe."
fi

echo "Iniciando la aplicación..."
streamlit run app.py --server.port $PORT