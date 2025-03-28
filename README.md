# Chatbot con LangChain, Gemini y BigQuery

Este proyecto implementa un chatbot utilizando **LangChain**, **Gemini** y **BigQuery** para responder preguntas sobre datos de taxis en Nueva York. Además, se ha integrado **ChromaDB** para mejorar la recuperación de información y optimizar el uso del modelo de Gemini.

---

## 📌 Requisitos Previos

### 1. Configuración del Entorno en GCP

#### 1.1 Creación del Proyecto en GCP
1. Ve a [Google Cloud Console](https://console.cloud.google.com/).
2. Navega a **IAM y Administración > Administrador de recursos**.
3. Crea un nuevo proyecto y asigna un nombre y una organización (si aplica).
4. Copia el ID del proyecto para utilizarlo más adelante.

#### 1.2 Habilitación de APIs Necesarias
Ejecuta los siguientes comandos en la terminal de Cloud Shell o en tu máquina local con `gcloud`:

```bash
gcloud services enable compute.googleapis.com \
    iam.googleapis.com \
    bigquery.googleapis.com \
    aiplatform.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com
```

#### 1.3 Configuración del Proyecto por Defecto
```bash
gcloud config set project TU_PROYECTO_ID
```

#### 1.4 Creación de la Cuenta de Servicio
```bash
gcloud iam service-accounts create mi-cuenta-servicio \
    --display-name "Cuenta de servicio para el chatbot"
```

#### 1.5 Asignación de Permisos
```bash
gcloud projects add-iam-policy-binding TU_PROYECTO_ID \
    --member="serviceAccount:mi-cuenta-servicio@TU_PROYECTO_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding TU_PROYECTO_ID \
    --member="serviceAccount:mi-cuenta-servicio@TU_PROYECTO_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

#### 1.6 Obtención de GOOGLE_APPLICATION_CREDENTIALS
```bash
gcloud iam service-accounts keys create key.json \
    --iam-account=mi-cuenta-servicio@TU_PROYECTO_ID.iam.gserviceaccount.com
```

Guarda el archivo `key.json` en un lugar seguro y configura la variable de entorno:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/ruta/a/tu/key.json"
```

#### 1.7 Obtención de GEMINI_API_KEY
1. Ve a [Google AI Studio](https://aistudio.google.com/).
2. Accede a la sección **API Keys**.
3. Genera una nueva clave y guárdala.
4. Configura la variable de entorno:
```bash
export GEMINI_API_KEY="tu-api-key"
```

#### 1.8 Creación de la Máquina Virtual
1. Ve a **Compute Engine > Instancias de VM** y crea una nueva instancia con:
   - Imagen: Ubuntu 22.04
   - Tipo de máquina: e2-medium (o superior)
   - Permitir tráfico HTTP y HTTPS
2. Conéctate a la VM con SSH:
```bash
gcloud compute ssh TU_VM --zone=us-central1-a
```

#### 1.9 Instalación de Dependencias en la VM
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-venv python3-pip git -y
```

### 2. Clonación del Repositorio y Configuración del Entorno Virtual

```bash
git clone <URL_DEL_REPO>
cd agente_vertexai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Implementación del Chatbot

### **1. Configurar el Código Base**

#### **Archivo `bot.py`**

```python
import os
import google.generativeai as genai
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from google.cloud import bigquery
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Configurar credenciales
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

client = bigquery.Client()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

# Funciones de consulta y respuesta
```

---

## 🖥️ **Ejecutar Localmente**

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Configuración de Firewall para Acceder a la Aplicación

```bash
gcloud compute firewall-rules create allow-streamlit \
    --allow=tcp:8501 --source-ranges=0.0.0.0/0 \
    --target-tags=streamlit
```

Accede a la aplicación en tu navegador con:
```
http://<IP_DE_TU_VM>:8501
```

---

## Posibles Errores y Soluciones

1️⃣ **Error: No module named 'langchain'**
✅ Solución: Asegúrate de instalar la versión correcta de LangChain.
```bash
pip install --upgrade langchain
```

2️⃣ **Error: ImportError en 'ChatGoogleGenerativeAI'**
✅ Solución: Revisa que estás usando una versión compatible de LangChain y Google Generative AI.
```bash
pip install --upgrade langchain google-generativeai
```

3️⃣ **Error: Puerto 8501 en uso**
✅ Solución: Mata los procesos en ejecución.
```bash
netstat -tulnp | grep 8501  # Verifica los puertos activos
kill $(lsof -t -i:8501)  # Cierra el proceso
```

4️⃣ **Error: Permisos insuficientes en GCP**
✅ Solución: Verifica tu archivo key.json y la variable de entorno.
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/ruta/a/tu/key.json"
```

---

## 📈 Monitoreo de Uso de Gemini

Para ver los llamados a la API de Gemini:

1. Ir a **Google Cloud Console** → IAM & Admin → Cuotas.
2. Buscar **Gemini API** y revisar los consumos.

---

## 🎯 Conclusión

Este chatbot combina **LLMs, recuperación semántica y análisis de datos en BigQuery** para proporcionar respuestas precisas sobre taxis en NY. 🚖💡

