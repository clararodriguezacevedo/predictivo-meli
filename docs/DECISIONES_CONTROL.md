# Decisiones y sus tests

Generado por `run_decisiones.py`.

Las comparaciones de modelos usan el t pareado corregido de Nadeau y Bengio sobre los 10 folds, con corrección de Holm por la cantidad de tests (su p ya está ajustado). La diferencia es con la pieza menos sin la pieza, en puntos de accuracy, casi siempre medida con LightGBM.

Veredictos: **respaldada** (al conservar, la pieza suma de forma significativa; al descartar, no se demostró que sume), **sin respaldo** (se conserva algo sin una mejora demostrada), **contradicha** (el test indica lo contrario), **sin test**.

Un p alto no prueba que una pieza no aporte: con 10 folds y tantos tests solo se detectan las diferencias grandes. Además, sacar una pieza sola subestima su aporte si otra lleva información parecida, por eso algunas se prueban juntas.

Resumen: {'respaldada': 42, 'sin respaldo': 21}

| Código   | Decisión                                                                                          | Prueba                                                             | p       | Veredicto    |
|:---------|:--------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|:--------|:-------------|
| D01      | Validar con folds al azar: train y test son intercambiables                                       | Mann-Whitney, AUC 0.497                                            | 0.122   | respaldada   |
| D02      | Descartar columnas constantes                                                                     | sin test: una constante no puede asociarse; no constantes: ninguna |         | respaldada   |
| D03      | Usar la ausencia de `official_store_id` como señal                                                | chi cuadrado, usados 46.7% si falta y 3.0% si está                 | < 0.001 | respaldada   |
| D03      | Usar la ausencia de `parent_item_id` como señal                                                   | chi cuadrado, usados 45.2% si falta y 46.6% si está                | 0.002   | respaldada   |
| D03      | Usar la ausencia de `video_id` como señal                                                         | chi cuadrado, usados 47.0% si falta y 23.0% si está                | < 0.001 | respaldada   |
| D03      | Usar la ausencia de `warranty` como señal                                                         | chi cuadrado, usados 51.6% si falta y 38.0% si está                | < 0.001 | respaldada   |
| D10      | Conservar: Tipo de publicación y estado                                                           | t corregido, +0.234 ± 0.042 puntos                                 | 0.018   | respaldada   |
| D11      | Conservar: Precio y moneda                                                                        | t corregido, +0.036 ± 0.034 puntos                                 | 1.000   | sin respaldo |
| D12      | Conservar: Stock y ventas                                                                         | t corregido, +0.008 ± 0.045 puntos                                 | 1.000   | sin respaldo |
| D14      | Conservar: Ubicación del vendedor                                                                 | t corregido, -0.009 ± 0.030 puntos                                 | 1.000   | sin respaldo |
| D15      | Conservar: Envío                                                                                  | t corregido, +0.031 ± 0.029 puntos                                 | 1.000   | sin respaldo |
| D16      | Conservar: Medios de pago                                                                         | t corregido, +0.019 ± 0.041 puntos                                 | 1.000   | sin respaldo |
| D17      | Conservar: Contenido de la publicación                                                            | t corregido, +0.074 ± 0.046 puntos                                 | 1.000   | sin respaldo |
| D18      | Conservar: Ficha técnica                                                                          | t corregido, +0.087 ± 0.039 puntos                                 | 1.000   | sin respaldo |
| D19      | Conservar: Features derivadas del item                                                            | t corregido, +0.589 ± 0.065 puntos                                 | < 0.001 | respaldada   |
| D20      | Conservar: Palabras clave en título y garantía                                                    | t corregido, +0.167 ± 0.042 puntos                                 | 0.138   | sin respaldo |
| D21      | Conservar: Perfil del vendedor                                                                    | t corregido, +0.066 ± 0.040 puntos                                 | 1.000   | sin respaldo |
| D22      | Conservar: Perfil de la categoría                                                                 | t corregido, +0.065 ± 0.035 puntos                                 | 1.000   | sin respaldo |
| D23      | Conservar: Perfil por palabra del título                                                          | t corregido, +0.034 ± 0.039 puntos                                 | 1.000   | sin respaldo |
| D24      | Conservar: Formato de las fotos                                                                   | t corregido, -0.014 ± 0.043 puntos                                 | 1.000   | sin respaldo |
| D25      | Conservar: Desfasaje en la duración                                                               | t corregido, +0.349 ± 0.067 puntos                                 | 0.026   | respaldada   |
| D26      | Conservar: Fechas agregadas después                                                               | t corregido, +0.032 ± 0.041 puntos                                 | 1.000   | sin respaldo |
| D27      | Conservar: Los tres perfiles juntos                                                               | t corregido, +0.242 ± 0.064 puntos                                 | 0.178   | sin respaldo |
| D30      | Conservar: Tasa de usados del vendedor                                                            | t corregido, +0.017 ± 0.031 puntos                                 | 1.000   | sin respaldo |
| D31      | Conservar: Tasa de usados de la categoría                                                         | t corregido, +0.006 ± 0.037 puntos                                 | 1.000   | sin respaldo |
| D32      | Conservar: Vecinos por título                                                                     | t corregido, +0.085 ± 0.068 puntos                                 | 1.000   | sin respaldo |
| D33      | Conservar: Tasas por vendedor cruzado con tipo, stock, garantía y título                          | t corregido, -0.033 ± 0.060 puntos                                 | 1.000   | sin respaldo |
| D34      | Conservar: Vecinos del mismo vendedor                                                             | t corregido, +0.019 ± 0.042 puntos                                 | 1.000   | sin respaldo |
| D35      | Conservar: Todo el target encoding junto                                                          | t corregido, +0.559 ± 0.076 puntos                                 | 0.002   | respaldada   |
| D36      | Conservar: Todas las piezas de efecto casi nulo juntas                                            | t corregido, +0.204 ± 0.059 puntos                                 | 0.280   | sin respaldo |
| D40      | Vecinos por título: k = 30 y no 10                                                                | t corregido, +0.016 ± 0.028 puntos                                 | 1.000   | respaldada   |
| D41      | Suavizado de las tasas: 2 y no 10                                                                 | t corregido, -0.005 ± 0.028 puntos                                 | 1.000   | respaldada   |
| D42      | Suavizado de las tasas: 2 y no 0.5                                                                | t corregido, -0.023 ± 0.034 puntos                                 | 1.000   | respaldada   |
| D50      | Usar solo LightGBM y no promediar cuatro modelos                                                  | t corregido, +0.038 ± 0.024 puntos                                 | 1.000   | respaldada   |
| D51      | No adoptar los hiperparámetros de la última búsqueda                                              | t corregido, +0.001 ± 0.025 puntos                                 | 1.000   | respaldada   |
| D52      | Cortar en 0.5 y no en 0.48, el mejor valor visto                                                  | McNemar, diferencia +0.019 puntos                                  | 0.182   | respaldada   |
| D53      | Conservar fotos, ubicación, tasa del vendedor y fechas: sacarlas juntas pierde más de 0.02 puntos | t corregido, +0.029 ± 0.035 puntos                                 | 1.000   | sin respaldo |
| D60      | Sacar el modelo de texto (en LightGBM; además fue pedido tuyo)                                    | t corregido, +0.041 ± 0.048 puntos                                 | 1.000   | respaldada   |
| D61      | Sacar las pseudo-etiquetas                                                                        | t corregido, +0.044 ± 0.034 puntos                                 | 1.000   | respaldada   |
| D62      | Sacar Random Forest frente a LightGBM                                                             | t corregido, -0.780 ± 0.087 puntos                                 | < 0.001 | respaldada   |
| D63      | Sacar Extra Trees frente a LightGBM                                                               | t corregido, -0.730 ± 0.090 puntos                                 | < 0.001 | respaldada   |
| D64      | Sacar regresión logística frente a LightGBM                                                       | t corregido, -1.060 ± 0.127 puntos                                 | < 0.001 | respaldada   |
| D65      | No usar red neuronal frente a LightGBM                                                            | t corregido, -0.769 ± 0.070 puntos                                 | < 0.001 | respaldada   |
| D66      | No usar kNN frente a LightGBM                                                                     | t corregido, -4.357 ± 0.128 puntos                                 | < 0.001 | respaldada   |
| D67      | No usar clustering frente a LightGBM                                                              | t corregido, -5.875 ± 0.092 puntos                                 | < 0.001 | respaldada   |
| D68      | No sumar kNN ni clustering como variables                                                         | t corregido, +0.016 ± 0.039 puntos                                 | 1.000   | respaldada   |
| D69      | Sacar las palabras clave nuevas                                                                   | t corregido, +0.000 ± 0.036 puntos                                 | 1.000   | respaldada   |
| D70      | Sacar el subdominio de la URL                                                                     | t corregido, +0.006 ± 0.040 puntos                                 | 1.000   | respaldada   |
| D71      | Sacar el consenso del vendedor                                                                    | t corregido, -0.004 ± 0.043 puntos                                 | 1.000   | respaldada   |
| D80      | No sumar URL contra título                                                                        | t corregido, -0.004 ± 0.040 puntos                                 | 1.000   | respaldada   |
| D81      | No sumar más datos de la ficha técnica                                                            | t corregido, +0.014 ± 0.051 puntos                                 | 1.000   | respaldada   |
| D82      | No sumar más datos de las variaciones                                                             | t corregido, -0.024 ± 0.027 puntos                                 | 1.000   | respaldada   |
| D83      | No sumar rastros del JSON crudo                                                                   | t corregido, -0.016 ± 0.038 puntos                                 | 1.000   | respaldada   |
| D84      | No sumar tasas por garantía, ciudad y categoría cruzada                                           | t corregido, -0.012 ± 0.048 puntos                                 | 1.000   | respaldada   |
| D85      | No limpiar etiquetas: sacar dudosas al 99%                                                        | t corregido, -0.064 ± 0.037 puntos                                 | 1.000   | respaldada   |
| D86      | No limpiar etiquetas: sacar dudosas al 95%                                                        | t corregido, -0.131 ± 0.057 puntos                                 | 1.000   | respaldada   |
| D87      | No limpiar etiquetas: sacar dudosas al 90%                                                        | t corregido, -0.161 ± 0.046 puntos                                 | 0.266   | respaldada   |
| D88      | No limpiar etiquetas: sacar dudosas al 80%                                                        | t corregido, -0.231 ± 0.064 puntos                                 | 0.229   | respaldada   |
| D89      | No limpiar etiquetas: invertir dudosas al 95%                                                     | t corregido, -0.154 ± 0.039 puntos                                 | 0.152   | respaldada   |
| D90      | No hacer stacking de los cuatro modelos                                                           | t corregido, -0.222 ± 0.052 puntos                                 | 0.093   | respaldada   |
| D91      | No entrenar un modelo por segmento de stock                                                       | t corregido, -0.101 ± 0.026 puntos                                 | 0.162   | respaldada   |
| D92      | No promediar varias semillas de LightGBM                                                          | t corregido, +0.018 ± 0.032 puntos                                 | 1.000   | respaldada   |
| D93      | 5 sub-folds y no 10 en el target encoding                                                         | t corregido, -0.009 ± 0.034 puntos                                 | 1.000   | respaldada   |
