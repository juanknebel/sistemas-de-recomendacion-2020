# Postulaciones Zonajobs - FCEN 2020

### Requisitos
1. Python 3.8
2. lightfm
3. pandas
4. numpy
5. tqdm
6. scikit-learn

### Directorios y archivos necesarios
1. Deben existir los siguientes directorios:
    1. ```data/01_raw/```
    2. ```data/02_intermediate/```
    3. ```data/07_model_output/```
2. Los archivos iniciales deben copiarse a:
    1. ```data/01_raw/avisos/avisos_detalle.csv```
    2. ```data/01_raw/avisos/avisos_online.csv```
    3. ```data/01_raw/postulacions/postulaciones_train.csv```
    4. ```data/01_raw/postulantes/postulantes_educacion.csv```
    5. ```data/01_raw/postulantes/postulantes_genero_edad.csv```
    6. ```data/01_raw/ejemplo_de_solucion.csv```

### Ejecución del etl
Antes de ejecutar el recomendador se debe ejecutar el etl de la siguiente manera
```bash
python -m etl.etl
```

Esto genera los nuevos arhivos que va a utilizar el recomendador.

### Ejecución del recomendador
Para obtener el archivo csv con las recomendacion ejecutar lo siguiente
```bash
python rank.py "identificador"
```
El mismo queda guardado en ```data/07_model_output/lightfm.csv```
