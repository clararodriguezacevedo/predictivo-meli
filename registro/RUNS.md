# Registro de corridas

Resumen generado desde `runs.jsonl`, que tiene el detalle completo de cada corrida (hiperparámetros exactos, accuracy por fold, features, commit).

## Validaciones (protocolo estándar, comparables entre sí)

| id                                  | modelo   |   accuracy |   error estándar | target encoding   |   features | nota                                                   | parámetros                                                                                                                                                                        |
|:------------------------------------|:---------|-----------:|-----------------:|:------------------|-----------:|:-------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 20261010-194243-validacion-lightgbm | lightgbm |     0.9429 |           0.0007 | sí                |        157 | mejor de 17 pruebas de búsqueda                        | n_estimators=1500, learning_rate=0.03431, num_leaves=149, min_child_samples=5, colsample_bytree=0.6875, subsample=0.8345, subsample_freq=1, reg_lambda=0.07827, reg_alpha=0.02924 |
| 20261010-195055-validacion-lightgbm | lightgbm |     0.9429 |           0.0007 | sí                |        162 | mejores parametros de la busqueda + vecinos por titulo | n_estimators=1500, learning_rate=0.03431, num_leaves=149, min_child_samples=5, colsample_bytree=0.6875, subsample=0.8345, subsample_freq=1, reg_lambda=0.07827, reg_alpha=0.02924 |
| 20261010-192941-validacion-lightgbm | lightgbm |     0.9428 |           0.0006 | sí                |        162 | agrega vecinos por titulo (k=30)                       | n_estimators=1000, learning_rate=0.02652, num_leaves=93, min_child_samples=18, colsample_bytree=0.7351, subsample=0.9818, subsample_freq=1, reg_lambda=0.2154, reg_alpha=0.8483   |
| 20261010-194004-validacion-xgboost  | xgboost  |     0.9427 |           0.0007 | sí                |        157 | mejor de 44 pruebas de búsqueda                        | n_estimators=1000, learning_rate=0.03136, max_depth=8, min_child_weight=0.971, colsample_bytree=0.4514, subsample=0.8149, reg_lambda=0.01763, reg_alpha=0.001278, gamma=0.0146    |
| 20261010-182854-validacion-lightgbm | lightgbm |     0.9425 |           0.0006 | sí                |        157 | mejor de 9 pruebas de búsqueda                         | n_estimators=1000, learning_rate=0.02652, num_leaves=93, min_child_samples=18, colsample_bytree=0.7351, subsample=0.9818, subsample_freq=1, reg_lambda=0.2154, reg_alpha=0.8483   |
| 20261010-182917-validacion-xgboost  | xgboost  |     0.9422 |           0.0006 | sí                |        157 | mejor de 14 pruebas de búsqueda                        | n_estimators=700, learning_rate=0.02735, max_depth=10, min_child_weight=0.6244, colsample_bytree=0.6001, subsample=0.8353, reg_lambda=0.05389, reg_alpha=0.002998, gamma=0.011    |
| 20261010-175245-validacion-xgboost  | xgboost  |     0.9421 |           0.0006 | sí                |        157 | parametros por defecto                                 | n_estimators=400, learning_rate=0.05, max_depth=8, colsample_bytree=0.5, subsample=0.8                                                                                            |
| 20261010-175236-validacion-lightgbm | lightgbm |     0.9419 |           0.0006 | sí                |        157 | parametros por defecto                                 | n_estimators=400, learning_rate=0.05, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1                                                                        |

## Submissions

| id                                  | modelo   | archivo                                          | validación de origen                |   accuracy de validación | leaderboard público   |
|:------------------------------------|:---------|:-------------------------------------------------|:------------------------------------|-------------------------:|:----------------------|
| 20261010-195207-submission-lightgbm | lightgbm | outputs/submissions/20261010-195207-lightgbm.csv | 20261010-195055-validacion-lightgbm |                   0.9429 | sin cargar            |

## Búsqueda de hiperparámetros

Protocolo más liviano que el estándar. Se muestran las 15 mejores pruebas de cada modelo.

### lightgbm (17 pruebas)

| id                                |   accuracy | parámetros                                                                                                                                                                        |
|:----------------------------------|-----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 20261010-185945-busqueda-lightgbm |     0.9419 | n_estimators=1500, learning_rate=0.03431, num_leaves=149, min_child_samples=5, colsample_bytree=0.6875, subsample=0.8345, subsample_freq=1, reg_lambda=0.07827, reg_alpha=0.02924 |
| 20261010-193059-busqueda-lightgbm |     0.9417 | n_estimators=1500, learning_rate=0.05272, num_leaves=96, min_child_samples=8, colsample_bytree=0.6337, subsample=0.9794, subsample_freq=1, reg_lambda=0.01837, reg_alpha=0.01165  |
| 20261010-192112-busqueda-lightgbm |     0.9415 | n_estimators=1400, learning_rate=0.04116, num_leaves=130, min_child_samples=5, colsample_bytree=0.5366, subsample=0.9101, subsample_freq=1, reg_lambda=0.1772, reg_alpha=0.006076 |
| 20261010-190903-busqueda-lightgbm |     0.9414 | n_estimators=1500, learning_rate=0.0432, num_leaves=149, min_child_samples=7, colsample_bytree=0.7131, subsample=0.7456, subsample_freq=1, reg_lambda=0.08802, reg_alpha=0.04834  |
| 20261010-184329-busqueda-lightgbm |     0.9414 | n_estimators=1400, learning_rate=0.04459, num_leaves=217, min_child_samples=7, colsample_bytree=0.7572, subsample=0.944, subsample_freq=1, reg_lambda=0.1729, reg_alpha=0.4489    |
| 20261010-180015-busqueda-lightgbm |     0.9413 | n_estimators=1000, learning_rate=0.02652, num_leaves=93, min_child_samples=18, colsample_bytree=0.7351, subsample=0.9818, subsample_freq=1, reg_lambda=0.2154, reg_alpha=0.8483   |
| 20261010-183819-busqueda-lightgbm |     0.9412 | n_estimators=1300, learning_rate=0.06504, num_leaves=93, min_child_samples=12, colsample_bytree=0.6145, subsample=0.9948, subsample_freq=1, reg_lambda=0.7203, reg_alpha=0.4405   |
| 20261010-185317-busqueda-lightgbm |     0.9412 | n_estimators=1200, learning_rate=0.01895, num_leaves=198, min_child_samples=5, colsample_bytree=0.7839, subsample=0.9971, subsample_freq=1, reg_lambda=0.1531, reg_alpha=0.1929   |
| 20261010-181238-busqueda-lightgbm |     0.9411 | n_estimators=1500, learning_rate=0.03325, num_leaves=48, min_child_samples=10, colsample_bytree=0.6645, subsample=0.7281, subsample_freq=1, reg_lambda=0.9474, reg_alpha=0.001174 |
| 20261010-181606-busqueda-lightgbm |     0.9407 | n_estimators=1100, learning_rate=0.04094, num_leaves=85, min_child_samples=84, colsample_bytree=0.6091, subsample=0.6798, subsample_freq=1, reg_lambda=0.3308, reg_alpha=0.3806   |
| 20261010-175516-busqueda-lightgbm |     0.9407 | n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=25, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=3.068, reg_alpha=0.1697              |
| 20261010-183352-busqueda-lightgbm |     0.9399 | n_estimators=1000, learning_rate=0.0519, num_leaves=82, min_child_samples=24, colsample_bytree=0.4542, subsample=0.8229, subsample_freq=1, reg_lambda=0.3323, reg_alpha=1.989     |
| 20261010-181819-busqueda-lightgbm |     0.9397 | n_estimators=800, learning_rate=0.09736, num_leaves=19, min_child_samples=9, colsample_bytree=0.2968, subsample=0.8266, subsample_freq=1, reg_lambda=0.07598, reg_alpha=0.05307   |
| 20261010-181716-busqueda-lightgbm |     0.9391 | n_estimators=300, learning_rate=0.04643, num_leaves=99, min_child_samples=9, colsample_bytree=0.2774, subsample=0.6577, subsample_freq=1, reg_lambda=0.1839, reg_alpha=0.1286     |
| 20261010-180336-busqueda-lightgbm |     0.9391 | n_estimators=900, learning_rate=0.03699, num_leaves=206, min_child_samples=6, colsample_bytree=0.2523, subsample=0.5101, subsample_freq=1, reg_lambda=7.855, reg_alpha=0.7558     |

### xgboost (44 pruebas)

| id                               |   accuracy | parámetros                                                                                                                                                                        |
|:---------------------------------|-----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 20261010-192639-busqueda-xgboost |     0.942  | n_estimators=1000, learning_rate=0.03136, max_depth=8, min_child_weight=0.971, colsample_bytree=0.4514, subsample=0.8149, reg_lambda=0.01763, reg_alpha=0.001278, gamma=0.0146    |
| 20261010-191432-busqueda-xgboost |     0.9418 | n_estimators=600, learning_rate=0.03173, max_depth=9, min_child_weight=0.6002, colsample_bytree=0.5818, subsample=0.8403, reg_lambda=0.2408, reg_alpha=0.01836, gamma=0.005009    |
| 20261010-190025-busqueda-xgboost |     0.9413 | n_estimators=700, learning_rate=0.03793, max_depth=7, min_child_weight=0.5818, colsample_bytree=0.5044, subsample=0.7957, reg_lambda=0.4074, reg_alpha=0.004021, gamma=0.006329   |
| 20261010-192859-busqueda-xgboost |     0.9412 | n_estimators=700, learning_rate=0.03256, max_depth=8, min_child_weight=0.7742, colsample_bytree=0.5253, subsample=0.7661, reg_lambda=0.4089, reg_alpha=0.0125, gamma=0.007134     |
| 20261010-192339-busqueda-xgboost |     0.9412 | n_estimators=900, learning_rate=0.02864, max_depth=8, min_child_weight=0.8147, colsample_bytree=0.4408, subsample=0.7972, reg_lambda=0.056, reg_alpha=0.001719, gamma=0.02053     |
| 20261010-184155-busqueda-xgboost |     0.9411 | n_estimators=600, learning_rate=0.02691, max_depth=10, min_child_weight=0.5625, colsample_bytree=0.4155, subsample=0.7423, reg_lambda=0.07632, reg_alpha=0.001409, gamma=0.009833 |
| 20261010-191201-busqueda-xgboost |     0.9411 | n_estimators=900, learning_rate=0.02491, max_depth=7, min_child_weight=0.8478, colsample_bytree=0.4296, subsample=0.7507, reg_lambda=0.1553, reg_alpha=0.001293, gamma=0.01598    |
| 20261010-180622-busqueda-xgboost |     0.941  | n_estimators=700, learning_rate=0.02735, max_depth=10, min_child_weight=0.6244, colsample_bytree=0.6001, subsample=0.8353, reg_lambda=0.05389, reg_alpha=0.002998, gamma=0.011    |
| 20261010-185402-busqueda-xgboost |     0.9409 | n_estimators=900, learning_rate=0.02658, max_depth=8, min_child_weight=0.6793, colsample_bytree=0.4129, subsample=0.7087, reg_lambda=0.1995, reg_alpha=0.0013, gamma=0.006216     |
| 20261010-192052-busqueda-xgboost |     0.9409 | n_estimators=800, learning_rate=0.04252, max_depth=6, min_child_weight=0.6231, colsample_bytree=0.3943, subsample=0.7589, reg_lambda=0.07962, reg_alpha=0.001681, gamma=0.02319   |
| 20261010-184014-busqueda-xgboost |     0.9409 | n_estimators=800, learning_rate=0.03139, max_depth=6, min_child_weight=5.417, colsample_bytree=0.5755, subsample=0.985, reg_lambda=4.604, reg_alpha=0.01553, gamma=0.008815       |
| 20261010-190441-busqueda-xgboost |     0.9409 | n_estimators=900, learning_rate=0.04027, max_depth=7, min_child_weight=0.9412, colsample_bytree=0.56, subsample=0.7523, reg_lambda=0.8041, reg_alpha=0.005312, gamma=0.006849     |
| 20261010-180057-busqueda-xgboost |     0.9409 | n_estimators=800, learning_rate=0.06033, max_depth=5, min_child_weight=5.299, colsample_bytree=0.286, subsample=0.9723, reg_lambda=0.6524, reg_alpha=0.03418, gamma=0.00747       |
| 20261010-190631-busqueda-xgboost |     0.9409 | n_estimators=600, learning_rate=0.02566, max_depth=8, min_child_weight=0.7455, colsample_bytree=0.3841, subsample=0.7464, reg_lambda=0.3282, reg_alpha=0.001439, gamma=0.009558   |
| 20261010-183846-busqueda-xgboost |     0.9408 | n_estimators=700, learning_rate=0.06365, max_depth=5, min_child_weight=6.928, colsample_bytree=0.2006, subsample=0.963, reg_lambda=4.606, reg_alpha=0.01721, gamma=0.007808       |
