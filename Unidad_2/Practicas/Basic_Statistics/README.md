### Algoritmos de:
* [Correlación](#Correlacion) 
* [Evaluación de la hipótesis](#Evaluación_de_la_hipótesis)
* [Summarizer](#Summarizer)  

#### Correlacion
### Que es el algoritmo Correlación?

Correlation calcula la matriz de correlación para el conjunto de datos de entrada de vectores utilizando el método especificado. La salida será un DataFrame que contiene la matriz de correlación de la columna de vectores.

###Como funciona: 
Importamos las bibliotecas y paquetes necesarios para cargar el programa.
* import org.apache.spark.ml.linalg.{Matrix, Vectors}
* import org.apache.spark.ml.stat.Correlation
* import org.apache.spark.sql.Row
* import org.apache.spark.sql.SparkSession

Creamos una instancia de la sesion de spark
* val spark = SparkSession.builder.appName ("CorrelationExample"). GetOrCreate ()

Realizamos una importacion de:
* import spark.implicits._
implicitsobjeto da conversiones implícitas para convertir objetos Scala (incl. DDR) en una Dataset, DataFrame, Columnso apoyar tales conversiones.
implicits es un objeto que se define dentro de SparkSession y, por lo tanto, requiere que cree una instancia de SparkSession primero antes de importar las implicitsconversiones.

Declaracion de los vectores que se utilizaran para este algoritmo las diferencias que existe  entre declarar un vector como disperso  o un vector denso es que el vector denso está respaldado por una matriz doble que representa sus valores de entrada, mientras que un vector disperso está respaldado por dos matrices paralelas: índices y valores.
* val data = Seq(
  Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
  Vectors.dense(4.0, 5.0, 0.0, 3.0),
  Vectors.dense(6.0, 7.0, 0.0, 8.0),
  Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
    )
Posteriormente creamos el dataframe y creemos un marco de datos a partir de él, de la siguiente manera:
* val df = data.map(Tuple1.apply).toDF("features")

Generamos nuestra Matriz de correlación de Pearson e imprimimos los resultados:
* val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
* println(s"Pearson correlation matrix:\n $coeff1")

Y por ultimo generamos nuestra Matriz de correlación de Spearman e imprimimos los resultados:
* val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
* println(s"Spearman correlation matrix:\n $coeff2")


#### Evaluación_de_la_hipótesis
### Que es el algoritmo Evaluación de la hipótesis?
La prueba de hipótesis es una herramienta poderosa en estadística para determinar si un resultado es estadísticamente significativo, si este resultado ocurrió por casualidad o no. spark.mlactualmente es compatible con Chi-cuadrado de Pearson ( χ2) pruebas de independencia.

### Como funciona:
Importamos las bibliotecas y paquetes necesarios para cargar el programa.
* import org.apache.spark.ml.linalg.{Vector, Vectors}
* import org.apache.spark.ml.stat.ChiSquareTest
* import org.apache.spark.sql.SparkSession

Creamos una instancia de la sesion de spark
* val spark = SparkSession.builder.appName("EvaluacionHipotesisExample").getOrCreate ()

Realizamos una importacion de:
* import spark.implicits._

implicitsobjeto da conversiones implícitas para convertir objetos Scala (incl. DDR) en una Dataset, DataFrame, Columnso apoyar tales conversiones.
implicits es un objeto que se define dentro de SparkSession y, por lo tanto, requiere que cree una instancia de SparkSession primero antes de importar las implicitsconversiones.

Se declaracion de los vectores densos está respaldado por una matriz doble que representa sus valores de entrada.

* val data = Seq(
  (0.0, Vectors.dense(0.5, 10.0)),
  (0.0, Vectors.dense(1.5, 20.0)),
  (1.0, Vectors.dense(1.5, 30.0)),
  (0.0, Vectors.dense(3.5, 30.0)),
  (0.0, Vectors.dense(3.5, 40.0)),
  (1.0, Vectors.dense(3.5, 40.0))
) 
 Convertimos de etiquetas a caracteristicas el dataframe
 * val df = data.toDF("label", "features")


Función de prueba de invocación de la clase ChiSquareTest:
Realice la prueba de independencia de Pearson para cada característica contra la etiqueta. Para cada característica, los pares (característica, etiqueta) se convierten en una matriz de contingencia para la cual se calcula la estadística Chi-cuadrado. Todos los valores de etiquetas y características deben ser categóricos.
La hipótesis nula es que la aparición de los resultados es estadísticamente independiente.

Parámetros:
dataset- DataFrame de etiquetas categóricas y características categóricas. Las características con valor real se tratarán como categóricas para cada valor distinto.

Devoluciones:
DataFrame que contiene el resultado de la prueba para cada característica contra la etiqueta. Este DataFrame contendrá una única fila con los siguientes campos: - pValues: Vector - degreesOfFreedom: Array[Int] - statistics: Vector Cada uno de estos campos tiene un valor por característica.
* val chi = ChiSquareTest.test(df, "features", "label").head

Finalmente imprimimos los resultados del algoritmo:
* println(s"pValues = ${chi.getAs[Vector](0)}") ---> Nombre de la columna de características en el conjunto de datos, de tipo Vector( VectorUDT) y  Nombre de la columna de etiqueta en el conjunto de datos, de cualquier tipo numérico
*  println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}") ---> grado de libertad en el que se puede desplazar
son el número de celdas que necesita completar antes, dados los totales en los márgenes, puede completar el resto de la cuadrícula utilizando una fórmula.
Si tiene un conjunto dado de totales para cada columna y fila, entonces no tiene libertad ilimitada al completar las celdas. 
* println(s"statistics ${chi.getAs[Vector](2)}")
Se tiene el 75% de posibilidades de encontrar una discrepancia entre las distribuciones observadas y esperadas que es al menos este extremo.

#### Summarizer
### Que es el algoritmo Sumarizacion?
El uso Summarizer para calcular la media y la varianza para una columna de vector del marco de datos de entrada, con y sin una columna de peso.

### Como funciona:
Importamos las bibliotecas y paquetes necesarios para cargar el programa.
* import org.apache.spark.ml.linalg.{Vector, Vectors}
* import org.apache.spark.ml.stat.Summarizer
* import org.apache.spark.sql.SparkSession

Creamos una instancia de la sesion de spark
* val spark = SparkSession.builder.appName ("CorrelationExample"). GetOrCreate ()

Realizamos una importacion de:
* import spark.implicits._
implicitsobjeto da conversiones implícitas para convertir objetos Scala (incl. DDR) en una Dataset, DataFrame, Columnso apoyar tales conversiones.
implicits es un objeto que se define dentro de SparkSession y, por lo tanto, requiere que cree una instancia de SparkSession primero antes de importar las implicitsconversiones.
* import Summarizer._
Herramientas para estadísticas vectorizadas en vectores MLlib.
Los métodos en este paquete proporcionan varias estadísticas para los vectores contenidos dentro de DataFrames.

Se declaracion de los vectores densos está respaldado por una matriz doble que representa sus valores de entrada.
val data = Seq(
  (Vectors.dense(2.0, 3.0, 5.0), 1.0),
  (Vectors.dense(4.0, 6.0, 7.0), 2.0)
  )

 creacion del dataframe y Convertimos de etiquetas a peso el dataframe
* val df = data.toDF("features", "weight")

Calcule la media y la varianza de las columnas del marco de datos usando weightCol el cual es Parámetro para el nombre de la columna de peso.
* val (meanVal, varianceVal) = df.select(metrics("mean", "variance").summary($"features", $"weight").as("summary")).select("summary.mean", "summary.variance").as[(Vector, Vector)].first()

Resultado 
* println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")


Calcule la media y la varianza de las columnas del marco de datos sin usar weightCol
* val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features")).as[(Vector, Vector)].first()

Resultados
* println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")

#### Authors: HERNANDEZ GARCIA RIGOBERTO
#### Titule: Basic Statistics
