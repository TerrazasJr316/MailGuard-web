#!/bin/bash

# build.sh
# Este script se ejecutará en Render para instalar las dependencias.

# Actualizar pip
pip install --upgrade pip

# Instalar las dependencias listadas en requirements.txt
pip install -r requirements.txt
