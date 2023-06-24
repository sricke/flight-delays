## Predicting Flight Delays
======
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
    │
    ├── datasets
    │   ├── dataset_SCL.csv        <- Dataset de información de vuelos y atrasos
    │   └── synthetic_features.csv <- Dataset generado de features sintéticas.
    │
    ├── jmeter            
    │	└── load_test.jmx <- Archivo para hacer una prueba de estrés a la API en JMeter.
    │
    ├── models            
    │	└── flight_delay.json <- Model de entramiento de XGBoost guardado.
    │
    ├── notebooks             
    │   ├── hyperparam.ipynb  <- Jupyter Notebook de búsqueda de hiperparámetros del modelo.
    │   └── preprocess.ipynb  <- Jupyter Notebook que contiene la exploración de datos y generación de features sintéticas..      
    │
    ├── src               
    │   ├── dataset.py    <- Código para cargar los datasets, formateo y split entre entrenamiento y test.
    │   └── train.ipynb   <- Código a cargo del piepeline de entrenamiento. Generación de modelo, entrenamiento y evaluación.
    │   ├── data           <- Scripts to download or generate data
    │
    ├── .gitignore 
    ├── app.py           <- Archivo base de la API y sus endpoints. 
    ├── Dockerfile       <- Archivo para crear un container de Docker para la API
    ├── README.md        <- README descripción del proyecto y respuesta a las preguntas de análisis.
    └── requirements.txt <- Listado de dependencias
------------

### Descripción de los pasos y análisis:

##### Exploración de los datos y generación de features

La descripción de los datos y su exploración se encuentra en el archivo `notebooks/preprocess.ipynb`. Pero algunas conclusiones:

1. La distribucion de los datos según tiempo de atraso se **concentra en los valores mínimos y máximos**. Es decir, la mayoría de los atrasos son de menos de 15 minutos o más de 45. Llama la atención la gran cantidad de atrasos de más de 45 minutos, pero esto se puede deber a fenómenos extremos más recurrentes como temporales o problemas con el avión. 
2. Si vemos la distribucion de probablidad de atraso segun el tipo de vuelo, vemos que los **vuelos de tipo nacional concentran una mayor probabilidad de atraso** por lo que seria un buen predictor.
3. Siguiendo con lo anterior, vemos que **LATAM y SKY** concentran la mayor cantidad de atrasos de vuelos nacionales.
4. A la vez, si graficamos la distribucion de **probabilidad de atraso segun el dia** vemos que existe una diferencia muy clara entre ellos. Por ejemplo, que es mucho menos probable un atraso un sabado que un viernes. Esto se puede relacionar con el nivel de flujo de pasajeros en los aeropuertos esos mismos dias. 
5. Si graficamos la presencia de datos por aerolinea, vemos que estan fuertemente dominados por, como es de esperar, entre **LATAM y Sky**. Es necesario considerar esta distribucion al particionar los datos entre train/test.

En base a lo anterior, se consideró el **tipo de vuelo, quien lo opera y la ciudad de origen y destino** como buenos predictores para el modelo. También la **fecha**, pero para esto se necesitó generar features sintéticas en base a lo socilitado:

1. Se obtuvo la feature **temporada alta**, si la fecha programada del vuelo está entre 15 Diciembre y 3 Marzo, o 15 Julio y 31 Julio, o 11 Septiembre y 30 Septiembre.
2. Se obtuvo la feature **periodo del dia**, si la hora programada del vuelo es mañana (entre 5:00 y 11:59), tarde (entre 12:00 y 18:59) o noche (entre 19:00 y 4:59).

Algunas consideraciones con respecto a la **fecha** y su aplicación del modelo:

1. Se decidió utilizar el **día nominal en vez de número del día** entendiendo que la primera entrega más información con respecto al atraso ya que el número del mes cambia durante los años. Además, se pudo ver en la exploración de los datos que existen diferentes distribuciones de probablidad de atraso según el día nominal. 
2. Se decidió **no utlizar el año** ya que la mayor parte del dataset es del año 2017. Solo 2 datos son del 2018.

Y finalmente algunas consideraciones extras con respecto a las demás features:

1. Se decidió **no utilizar el número de operación del vuelo** ya que en general se entiende que las aerolíneas asignan un número de operación único entre ciudad de origen y destino, por lo que ya se le entrega suficiente información al modelo con la **ciudad de origen y destino**. Tampoco se utilizó el número programado ya que en la mayoría de los casos corresponden a lo mismo (se comprobó en el dataset).  
2. No se utlizaron las siglas de origen y destino de las ciudades del vuelo programado, sino que su nombre, entendiendo que el mapeo es 1:1. Tampooco las siglas de las ciudades de operación, ya que a lo menos en este dataset corresponden a lo mismo (sólo unas pocas excepciones en el caso de destino).
Ver **anexo** en el `notebooks/preprocess.ipynb` para comprobar.

Finalmente se quedó con las **8** features DIANOM, MES, TIPOVUELO, OPERA, SIGLAORI, SIGLADES, Temp-A, Per-D,ATRASADO

Con respecto a la variable objetivo, predecir atraso, se decidio **NO tomar todas todos los atrasos mayores a 0 minutos** por cómo se distribuyen los datos. Ya que los rangos inferiores concentran un alto número de datos, se decidió tomar un mínimo. Ya que se pidió considerar un atraso mínimo como menor a 15 minutos se decidió tomar este valor. Es decir, **si un atraso es mayor a 15 minutos se considera como atrasado, sino no**. A paritr de esto se obtuvo la variable binaria de atraso. 

