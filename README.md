# Implementación de nuevas tecnologías como el machine learning y el análisis de datos para el mercado bancario

## 1 Introducción

Los alcances de la inteligencia artificial hoy por hoy llegado a todas las áreas del mercado que por medio de los algoritmos machine learning que, realizando tareas de clasificación, predicción y agrupamiento, podemos encontrar patrones, tendencias y hasta comportamientos del mercado hacia las ventas, preferencias, segmentación, desarrollo de producto, demanda, y demás [1], [2]. Para este tema, la minería de datos que se encarga del manejo de estas técnicas y algunas otras relacionadas ha desarrollado, implementado y evaluado los algoritmos de tal manera que se usen dentro del mercado, para usos de este artículo, se desarrollara el proceso de minería de datos con una base de datos de un banco portugués, el cual lanzó una campaña de llamadas para la suscripción de depósitos a largo plazo a causa de una disminución de ingresos, para esto se desarrollaron algoritmos de clasificación supervisada, para determinar que perfiles si se suscribirían al producto y no se suscribirían teniendo en cuenta las ventajas de la minería de datos para el desarrollo de este problema, además se desarrolló un algoritmo clustering no supervisado para determinar quiénes si se suscribirían. Dentro de este desarrollo también se incluyó el análisis exploratorio de datos con base en gráficos [3]–[5]. La base de datos contiene información de duración de llamadas, productos abiertos con el banco, estado civil, edad, último contacto, educación, ocupación, frecuencia y deuda, con base a esta información de cada cliente se clasificó y se agrupó.

**Palabras clave: Algoritmos, machine learning, bank marketing, classification, clustering.**

## 2 Trabajo relacionado

Como se relacionó en la introducción, podemos ver que al problema a desarrollar es la disminución de ingresos del banco, por lo tanto se propuso la solución de aplicar modelos de predicción para tener una clasificación de personas que se suscribirían al depósito a largo plazo, teniendo en cuenta las ventajas que tiene el manejo de datos, la minería y las nuevas opciones para mejorar el marketing de un banco como se evidencia en la literatura evaluada en el banco UK, para esto se utilizó un algoritmo de clasificación dentro de los cuales se seleccionaron y evaluaron Naive bayes, KNN clasifier, Support Vector Machine [1]–[3]
Además de la clasificación, también se hizo un clustering no supervisado con el método K prototipe para agrupan las características que definen que una persona se suscribiría al depósito a largo plazo. Las ventajas del machine learning es que se puede utilizar en cualquier área, no solo la bancaria, para el caso de la literatura que se buscó para usos de este artículo, tenemos el uso de KNN para predecir posibles problemas durante construcciones, teniendo en cuenta anteriores problemas en anteriores construcciones de los estados unidos, con el fin de optimizar el beneficio evitando los problemas [6]. Además de esto también tenemos un modelo de clasificación de patologías en las cuerdas vocales usando Naive bayes, teniendo en cuenta el tono de voz, se analiza y predice si el paciente tiene alta probabilidad de contraer una enfermedad vocal, también se tienen factores en cuenta como la ocupación debido a que algunas usan las cuerdas vocales más que en otros como en el call center [7].

## 3 Definición del problema y algoritmo

### 3.1 Problema

En este proyecto se está trabajando con un banco portugués, el cual presenta una disminución en los ingresos, el banco está buscando conocer la razón del problema y saber que acciones tomar. Después de realizar una investigación encontraron que la causa raíz de su problema era que sus clientes no estaban invirtiendo en depósitos a largo plazo en la cantidad que se requiere. Habiendo encontrado esto al banco le gustaría identificar a los clientes que tienen actualmente de forma que se pueda identificar cuales presentan una mayor posibilidad de suscribirse a un depósito de largo plazo y así enfocar las próximas campañas de marketing en dichos clientes.
Los datos que se presentan en el dataset se basan en campañas de marketing del banco las cuales se hicieron en llamadas telefónicas y en ciertas ocasiones presenciales.

### 3.2 Algoritmos

Los algoritmos utilizados para clasificar a los clientes como aptos a suscribirse o no fueron ‘KNN neighbors’, ‘Naive Bayes’ y ‘Support Vector Machine’, y se evaluó el que tuviera mejores métricas haciendo distintas iteraciones y aplicando técnicas para mejorar los modelos.

El ‘KNN neighbors’ es un modelo predictivo que recorre los datos y calcula la distancia entre ellos, una vez la calcula, este busca los k vecinos más cercanos, dependiendo de la distancia y entre el grupo de los 5 más cercanos, el algoritmo escoge la clase más común dentro del grupo y la asigna al dato buscado, esta es una herramienta que no solo se usa para la clasificación, sino que también se usa para imputación. Adicionalmente se aplicó la técnica ‘cross-validation’ para mejorar este modelo el cual itera con parámetros de manera aleatoria y selecciona la mejor combinación [1], [6].

Además, se usó el algoritmo ‘Naive Bayes’, el cual tiene varias variantes, dependiendo del tipo de dato que se esté utilizando se utiliza una distribución de probabilidad diferente, entre estas variantes se encuentran: Gaussiana, Bernoulli, y Multinomial. La Gaussiana utiliza una distribución de gauss especial para datos numéricos, la Bernoulli para datos boléanos y la multinomial para datos discretos. Este algoritmo se basa en la probabilidad condicional y el teorema de bayes para identificar la clase a la que pertenece con mayor probabilidad de que ocurra [1], [7].

El último algoritmo de clasificación que se usó fue el ‘SVM’ el cual se basa en trazar vectores para separar los datos de una clase y los datos de otra, obteniendo finalmente el un hiperplano que maximice el margen entre cada clase y de esta manera se clasifique dependiendo de su ubicación con respecto al hiperplano. Este algoritmo al igual que los anteriores dos, tiene la técnica ‘Cross-Validation’ [1], [8].

Por último, en la aplicación de algoritmos de clustering se hizo uso de ‘K-prototypes’, este algoritmo tiene como objetivo agrupar el conjunto de datos en un número K de grupos minimizando la función de costo. Una de las principales características de ‘K-prototypes’ es que a diferencia de otros algoritmos de clustering tales como K-means, este se puede aplicar a datasets que contengan variables categóricas mixtas.[9]

## 4 Evaluación experimental

## 4.1 Data

El data set utilizado está relacionado con el mundo bancario, específicamente los datos provienen de un banco de Portugal, este data set como se mencionó anteriormente muestra los datos de los clientes del banco, y la información recopilada sobre los clientes a través de las campañas de marketing realizadas. El data set contiene 21 columnas y 32950 filas; cada uno de los conjuntos de datos están ordenados por fecha. El conjunto de datos tiene como objetivo, lograr la clasificación de los clientes, prediciendo si estos se suscribirán en un depósito a largo plazo.

Las características del data-set se presentan a continuación en la Tabla 1:

## Diagrama ER

![Alt text](https://github.com/santicar1809/zuber/blob/master/datasets/238116676-2edba3f3-131c-40eb-b0d0-273d6213d7db.png)