# MeliChallenge - FCEN 2020

### Requisitos
1. Python 3.8
2. surprise
3. pandas
4. numpy
5. tqdm

### Directorios y archivos necesarios
1. Deben existir los siguientes directorios:
    1. ```data/01_raw/```
    2. ```data/02_intermediate/```
    3. ```data/03_primary/```
    5. ```data/07_model_output/```
2. Los archivos iniciales deben copiarse a:
    1. ```data/01_raw/item_data.jl```
    2. ```data/01_raw/test_dataset.jl```
    3. ```data/01_raw/train_dataset.jl```

### Ejecución del etl
Antes de ejecutar el recomendador se debe ejecutar el etl de la siguiente manera
```bash
python etl.py
```

Esto genera los nuevos arhivos que va a utilizar el recomendador.

### Ejecución del recomendador
Para obtener el archivo csv con las recomendacion ejecutar la jupyter notebook
que se encuentra con el nombre de ```challenge_meli.ipynb```.
El mismo queda guardado en ```data/07_model_output/submission.csv```
