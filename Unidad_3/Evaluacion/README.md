Objetivo: El objetivo de este examen es tratar de agrupar los clientes de regiones específicas de un distribuidor al mayoreo. Esto en base a las ventas de algunas categorías de productos. 
Las fuente de datos se encuentra en el repositorio: https://github.com/jcromerohdz/BigData/blob/master/Spark_clustering/Whole sale customersdata.csv 


#### Que es K-means?

### El algoritmo K-means, creado por MacQueen en 1967[1] es el algoritmo de clustering más conocido y utilizado ya que es de muy simple aplicación y eficaz. Sigue un procedimiento simple de clasificación de un conjunto de objetos en un determinado número K de clusters, K determinado a priori. El nombre de K-means viene porque representa cada uno de los clusters por la media (o  media ponderada) de sus puntos, es decir, por su centroide. 

### El algoritmo del K-means se realiza en 4 etapas:

### Inicialización: Se definen un conjunto de objetos a particionar, el número de grupos y un centroide por cada grupo. Algunas implementaciones del algoritmo estándar determinan los centroides iniciales de forma aleatoria; mientras que algunos otros procesan los datos y determinar los centroides mediante de cálculos.
### Clasificación: Para cada objeto de la base de datos, se calcula su distancia a cada centroide, se determina el centroide más cercano, y el objeto es incorporado al grupo relacionado con ese centroide.
### Cálculo de centroides: Para cada grupo generado en el paso anterior se vuelve a calcular su centroide.
### Condición de convergencia: Se han usado varias condiciones de convergencia, de las cuales las más utilizadas son las siguientes: converger cuando alcanza un número de iteraciones dado, converger cuando no existe un intercambio de objetos entre los grupos, o converger cuando la diferencia entre los centroides de dos iteraciones consecutivas es más pequeño que un umbral dado. Si la condición de convergencia no se satisface, se repiten los pasos dos, tres y cuatro del algoritmo. 

### Caracterısticas de K- means 

### Las reglas de clasificación por vecindad están basadas en la búsqueda en un conjunto de prototipos de los k prototipos más cercanos al patrón a clasificar.
### No hay un modelo global asociado a los conceptos a aprender.
### Las predicciones se realizan basándose en los ejemplos más parecidos al que hay que predecir.
### El coste del aprendizaje es 0, todo el coste pasa al cálculo de la predicción.
### Se conoce como mecanismo de aprendizaje perezoso (lazy learning).
### Debemos especificar una m´etrica para poder medir la proximidad. Suele utilizarse por razones computacionales la distancia Euclıdea, para este fın.
### Denominaremos conjunto de referencia (y lo notaremos por R) al conjunto de prototipos

1. Importar una simple sesión Spark. 
            
            import org.apache.spark.sql.SparkSession

2. Utilice las lineas de código para minimizar errores 
            
            import org.apache.log4j._
            Logger.getLogger("org").setLevel(Level.ERROR)

3. Cree una instancia de la sesión Spark 
            
            val spark = SparkSession.builder().getOrCreate()

4. Importar la librería de Kmeans para el algoritmo de agrupamiento. 
            
            import org.apache.spark.ml.clustering.KMeans

5. Carga el dataset de Wholesale Customers Data 
           
            val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Wholesale_customers_data.csv")
            //Visualizacion de como vienen los datos.
            data.show()
            //Esquema del dataset
            data.printSchema()

6. Seleccione las siguientes columnas: Fres, Milk, Grocery, Frozen, Detergents_Paper, 
            
            Delicassen y llamar a este conjunto feature_data 
            val feature_data = data.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper",$"Delicassen")
            // mostrar las características seleccionadas de nuestro dataset
            feature_data.show()

7. Importar Vector Assembler y Vector 
            
            import org.apache.spark.ml.feature.VectorAssembler
            import org.apache.spark.ml.linalg.Vectors

8. Crea un nuevo objeto Vector Assembler para las columnas de caracteristicas como un 
            
            conjunto de entrada, recordando que no hay etiquetas 
            val Assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")

9. Utilice el objeto assembler para transformar feature_data 
            
            // Creación de nuestra variable de entrenamiento que transforma nuestra características de nuestro dataset.
            val Training= Assembler.transform(feature_data)

10. Crear un modelo Kmeans con K=3 

            val kmeans = new KMeans().setK(3).setSeed(12345L)
            val model = kmeans.fit(Traning)

11.Evalúe los grupos utilizando 

            val WSSSE = model.computeCost(Traning)
            println(s"Within Set Sum of Squared Errors = $WSSSE")

            println("Cluster Centers: ")
            model.clusterCenters.foreach(println)

12. ¿Cuáles son los nombres de las columnas?

            scala> feature_data.columns
            res21: Array[String] = Array(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen)
### Resultado
            scala> :load Kmeans.scala
            Loading Kmeans.scala...
            import org.apache.spark.sql.SparkSession
            import org.apache.log4j._
            spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@4e77f29e
            import org.apache.spark.ml.clustering.KMeans
            data: org.apache.spark.sql.DataFrame = [Channel: int, Region: int ... 6 more fields]
            +-------+------+-----+-----+-------+------+----------------+----------+
            |Channel|Region|Fresh| Milk|Grocery|Frozen|Detergents_Paper|Delicassen|
            +-------+------+-----+-----+-------+------+----------------+----------+
            |      2|     3|12669| 9656|   7561|   214|            2674|      1338|
            |      2|     3| 7057| 9810|   9568|  1762|            3293|      1776|
            |      2|     3| 6353| 8808|   7684|  2405|            3516|      7844|
            |      1|     3|13265| 1196|   4221|  6404|             507|      1788|
            |      2|     3|22615| 5410|   7198|  3915|            1777|      5185|
            |      2|     3| 9413| 8259|   5126|   666|            1795|      1451|
            |      2|     3|12126| 3199|   6975|   480|            3140|       545|
            |      2|     3| 7579| 4956|   9426|  1669|            3321|      2566|
            |      1|     3| 5963| 3648|   6192|   425|            1716|       750|
            |      2|     3| 6006|11093|  18881|  1159|            7425|      2098|
            |      2|     3| 3366| 5403|  12974|  4400|            5977|      1744|
            |      2|     3|13146| 1124|   4523|  1420|             549|       497|
            |      2|     3|31714|12319|  11757|   287|            3881|      2931|
            |      2|     3|21217| 6208|  14982|  3095|            6707|       602|
            |      2|     3|24653| 9465|  12091|   294|            5058|      2168|
            |      1|     3|10253| 1114|   3821|   397|             964|       412|
            |      2|     3| 1020| 8816|  12121|   134|            4508|      1080|
            |      1|     3| 5876| 6157|   2933|   839|             370|      4478|
            |      2|     3|18601| 6327|  10099|  2205|            2767|      3181|
            |      1|     3| 7780| 2495|   9464|   669|            2518|       501|
            +-------+------+-----+-----+-------+------+----------------+----------+
            only showing top 20 rows

            root
            |-- Channel: integer (nullable = true)
            |-- Region: integer (nullable = true)
            |-- Fresh: integer (nullable = true)
            |-- Milk: integer (nullable = true)
            |-- Grocery: integer (nullable = true)
            |-- Frozen: integer (nullable = true)
            |-- Detergents_Paper: integer (nullable = true)
            |-- Delicassen: integer (nullable = true)

            feature_data: org.apache.spark.sql.DataFrame = [Fresh: int, Milk: int ... 4 more fields]
            +-----+-----+-------+------+----------------+----------+
            |Fresh| Milk|Grocery|Frozen|Detergents_Paper|Delicassen|
            +-----+-----+-------+------+----------------+----------+
            |12669| 9656|   7561|   214|            2674|      1338|
            | 7057| 9810|   9568|  1762|            3293|      1776|
            | 6353| 8808|   7684|  2405|            3516|      7844|
            |13265| 1196|   4221|  6404|             507|      1788|
            |22615| 5410|   7198|  3915|            1777|      5185|
            | 9413| 8259|   5126|   666|            1795|      1451|
            |12126| 3199|   6975|   480|            3140|       545|
            | 7579| 4956|   9426|  1669|            3321|      2566|
            | 5963| 3648|   6192|   425|            1716|       750|
            | 6006|11093|  18881|  1159|            7425|      2098|
            | 3366| 5403|  12974|  4400|            5977|      1744|
            |13146| 1124|   4523|  1420|             549|       497|
            |31714|12319|  11757|   287|            3881|      2931|
            |21217| 6208|  14982|  3095|            6707|       602|
            |24653| 9465|  12091|   294|            5058|      2168|
            |10253| 1114|   3821|   397|             964|       412|
            | 1020| 8816|  12121|   134|            4508|      1080|
            | 5876| 6157|   2933|   839|             370|      4478|
            |18601| 6327|  10099|  2205|            2767|      3181|
            | 7780| 2495|   9464|   669|            2518|       501|
            +-----+-----+-------+------+----------------+----------+
            only showing top 20 rows

            import org.apache.spark.ml.feature.VectorAssembler
            import org.apache.spark.ml.linalg.Vectors
            Assembler: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_0cb9d9c63e0b
            Traning: org.apache.spark.sql.DataFrame = [Fresh: int, Milk: int ... 5 more fields]
            kmeans: org.apache.spark.ml.clustering.KMeans = kmeans_29b95e7227e8
            19/12/01 20:26:36 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
            19/12/01 20:26:36 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
            model: org.apache.spark.ml.clustering.KMeansModel = kmeans_29b95e7227e8         
            warning: there was one deprecation warning; re-run with -deprecation for details
            WSSSE: Double = 8.095172370767671E10
            Within Set Sum of Squared Errors = 8.095172370767671E10
            Cluster Centers: 
            [7993.574780058651,4196.803519061584,5837.4926686217,2546.624633431085,2016.2873900293255,1151.4193548387098]
            [9928.18918918919,21513.081081081084,30993.486486486487,2960.4324324324325,13996.594594594595,3772.3243243243246]
            [35273.854838709674,5213.919354838709,5826.096774193548,6027.6612903225805,1006.9193548387096,2237.6290322580644]
