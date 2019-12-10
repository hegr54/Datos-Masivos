        Objetivo: El objetivo de este examen es tratar de agrupar los clientes de regiones específicas de un distribuidor al mayoreo. Esto en base a las ventas de algunas categorías de productos. 
        Las fuente de datos se encuentra en el repositorio: https://github.com/jcromerohdz/BigData/blob/master/Spark_clustering/Whole sale customersdata.csv 
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
