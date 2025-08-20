#!/bin/bash

# Pega aquí la URL que copiaste de GitHub Releases
DATASET_URL="https://drive.google.com/drive/folders/1FhUnL1ztagpWdLj04n0Tb0I9ZNm_ZF2s?usp=sharing"
# 1. Revisa si la carpeta del dataset ya existe.
#    Esto evita volver a descargarla si el servicio solo se reinicia.
if [ ! -d "ALERT" ]; then
    echo "Carpeta ALERT no encontrada. Descargando dataset..."

    # 2. Descarga el archivo .zip usando la URL.
    #    La opción -L es importante para que siga las redirecciones de GitHub.
    curl -L $DATASET_URL -o ALERT.zip

    # 3. Descomprime el archivo.
    echo "Descomprimiendo dataset..."
    unzip ALERT.zip

    # 4. (Opcional) Limpia el archivo .zip para ahorrar espacio.
    rm ALERT.zip

    echo "Dataset listo."
else
    echo "La carpeta ALERT ya existe. Saltando descarga."
fi

# 5. Finalmente, ejecuta la aplicación de Streamlit.
#    Este comando le dice a Streamlit que se ejecute en el puerto que Render le asigne.
streamlit run app.py --server.port $PORT