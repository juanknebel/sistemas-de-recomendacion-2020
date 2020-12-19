# Predicción de opiniones de libros - FCEN 2020

### Requisitos
1. Python 3.8
2. surprise
3. pandas
4. numpy
5. pickle

### Directorios y archivos necesarios
1. Deben existir los siguientes directorios:
    1. ```data/01_raw/```
    2. ```data/02_intermediate/```
    3. ```data/03_primary/```
    3. ```data/06_models/```
    5. ```data/07_model_output/```
2. Los archivos iniciales deben copiarse a:
    1. ```data/01_raw/ejemplo_solucion.csv```
    2. ```data/01_raw/libros.csv```
    3. ```data/01_raw/opiniones_test.csv```
    4. ```data/01_raw/opiniones_train.csv```
    5. ```data/01_raw/usuarios.csv```

### Ejecución del etl
Antes de ejecutar el recomendador se debe ejecutar el etl de la siguiente manera
```bash
python -m etl.etl
```

Esto genera los nuevos arhivos que va a utilizar el recomendador.

### Ejecución del recomendador
Para obtener el archivo csv con las recomendacion ejecutar lo siguiente
```bash
python predict_2.py "identificador"
```
El mismo queda guardado en ```data/07_model_output/all_identificador.csv```
