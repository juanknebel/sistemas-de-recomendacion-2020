Algoritmo | RMSE | Notas | Archivos usados
- | - | - | -
SVD | 1.584 | NA | raw
KNN | 1.683 | NA | raw
LGBM | 1.555 | Con dos columnas de modelos de SVD y KNN |  02_intermediate/opiniones_train_opiniones_modelos.csv y 02_intermediate/opiniones_test_opiniones_modelos.csv
LGBM | 1.674 | Con genero y anio | 02_intermediate/opiniones_test_modelos_medias.csv y 02_intermediate/02_intermediate/opiniones_train_modelos_medias.csv
LGBM | 1.323 | Con promedios -> esta overfitteando | 02_intermediate/opiniones_test_modelos_medias.csv y 02_intermediate/02_intermediate/opiniones_train_modelos_medias.csv
LGBM | 1.674 | Con la cantidad de opiniones de los usuarios | 02_intermediate/opiniones_train_opiniones_1.csv y 02_intermediate/opiniones_test_opiniones_2.csv