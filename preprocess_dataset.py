import os
import pickle
from model_trainer import Parser, parse_index # Reutilizamos el código que ya funciona

# --- IMPORTANTE ---
# Asegúrate de que la carpeta ALERT esté en el mismo directorio que este script
# cuando lo ejecutes localmente.

print("Iniciando el preprocesamiento del dataset...")
print("Esto puede tardar varios minutos, por favor ten paciencia.")

# Ruta al índice del dataset completo en tu máquina local
LOCAL_INDEX_PATH = "ALERT/datasets/datasets/trec07p/full/index"

# Contamos cuántos correos hay en total
with open(LOCAL_INDEX_PATH) as f:
    num_total_emails = len(f.readlines())

print(f"Se encontraron {num_total_emails} correos en el índice.")

# Usamos las funciones de model_trainer para procesar TODOS los correos
p = Parser()
indexes = parse_index(LOCAL_INDEX_PATH, num_total_emails)

X_full = []
y_full = []

for i, index in enumerate(indexes):
    pmail = p.parse(index["email_path"])
    if pmail:
        full_text = " ".join(pmail['subject']) + " " + " ".join(pmail['body'])
        X_full.append(full_text)
        y_full.append(index["label"])

    # Imprimir progreso para saber que no se ha colgado
    if (i + 1) % 1000 == 0:
        print(f"Procesados {i+1}/{num_total_emails} correos...")

# Guardamos los datos procesados en un único archivo binario con pickle
output_filename = 'preprocessed_spam_data.pkl'
with open(output_filename, 'wb') as f:
    pickle.dump((X_full, y_full), f)

print("-" * 50)
print(f"¡Éxito! El dataset ha sido procesado y guardado en '{output_filename}'")
print(f"Tamaño del archivo final: {os.path.getsize(output_filename) / (1024*1024):.2f} MB")
print("Ahora, sube este archivo a una nueva Release en GitHub.")