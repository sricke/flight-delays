## Predicting Flight Delays
==============================
Pasos para reproducir:

### Entrenar el modelo:

1. Mover el archivo `dataset_SCL.csv` a la carpeta datasets.
2. Correr el notebook ubicado en `notebooks/preprocess.ipynb` para generar el archivo `synthetic_features.csv`.
3. Instalar las dependencias utilizando **pip**: `pip install --no-cache-dir --upgrade -r /code/requirements.txt`.
4. Correr el arvico de entrenamiento: `python src/train.py`

### Organización del proyecto
------------
    ├── .github
    │   └── workflows
    │       └── gcp.yml <- Workflow de github actions para CI/CD de deploy de la API en Google Cloud Platform.
    ├── datasets
    │   ├── dataset_SCL.csv        <- Dataset de información de vuelos y atrasos
    │   └── synthetic_features.csv <- Dataset generado de features sintéticas.
    ├── jmeter            
    │	└── load_test.jmx <- Archivo para hacer una prueba de estrés a la API en JMeter.
    ├── models            
    │	└── flight_delay.json <- Model de entramiento de XGBoost guardado.
    ├── notebooks             
    │   ├── hyperparam.ipynb  <- Jupyter Notebook de búsqueda de hiperparámetros del modelo.
    │   └── preprocess.ipynb  <- Jupyter Notebook que contiene la exploración de datos y generación de features sintéticas..      
    ├── src               
    │   ├── dataset.py    <- Código para cargar los datasets, formateo y split entre entrenamiento y test.
    │   └── train.ipynb   <- Código a cargo del piepeline de entrenamiento. Generación de modelo, entrenamiento y evaluación.
    │   ├── data           <- Scripts to download or generate data
    ├── .gitignore 
    ├── app.py           <- Archivo base de la API y sus endpoints. 
    ├── Dockerfile       <- Archivo para crear un container de Docker para la API
    ├── README.md        <- README descripción del proyecto y respuesta a las preguntas de análisis.
    └── requirements.txt <- Listado de dependencias
------------