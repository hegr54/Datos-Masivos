            //////////////////////////////////////////////
            // Proyecto de regresion logistica //////////////
            ////////////////////////////////////////////
            //////////////////////////////////////////////////////////
            // Complete las siguientes tareas que estan comentas ////
            /////////////////////////////////////////////////////////



            ////////////////////////
            /// Tome los datos //////
            //////////////////////

            // Importacion de las librerias y apis de Logistic Regression
            import org.apache.spark.ml.classification.LogisticRegression
            //Importacion de la libreria de SparkSeccion
            import org.apache.spark.sql.SparkSession
            // Si requerimos la utilizacion del codigo de  Error reporting lo implementamos aunque regularmente es opcional esta accion.

            import org.apache.log4j._
            Logger.getLogger("org").setLevel(Level.ERROR)
            // Creamos la secion de spark. 
            val spark = SparkSession.builder().getOrCreate()
            // Utilice Spark para leer el archivo que se encuentra en un formato csv llamado Advertising.csv

            val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
            //Visualizacion de los datos de nustro dataset y analizamos como nos son precentados.
            data.show()
            /*data: org.apache.spark.sql.DataFrame = [Daily Time Spent on Site: double, Age: int ... 8 more fields]
            +------------------------+---+-----------+--------------------+--------------------+-----------------+----+--------------------+-------------------+-------------+
            |Daily Time Spent on Site|Age|Area Income|Daily Internet Usage|       Ad Topic Line|             City|Male|             Country|          Timestamp|Clicked on Ad|
            +------------------------+---+-----------+--------------------+--------------------+-----------------+----+--------------------+-------------------+-------------+
            |                   68.95| 35|    61833.9|              256.09|Cloned 5thgenerat...|      Wrightburgh|   0|             Tunisia|2016-03-27 00:53:11|            0|
            |                   80.23| 31|   68441.85|              193.77|Monitored nationa...|        West Jodi|   1|               Nauru|2016-04-04 01:39:02|            0|
            |                   69.47| 26|   59785.94|               236.5|Organic bottom-li...|         Davidton|   0|          San Marino|2016-03-13 20:35:42|            0|
            |                   74.15| 29|   54806.18|              245.89|Triple-buffered r...|   West Terrifurt|   1|               Italy|2016-01-10 02:31:19|            0|
            |                   68.37| 35|   73889.99|              225.58|Robust logistical...|     South Manuel|   0|             Iceland|2016-06-03 03:36:18|            0|
            |                   59.99| 23|   59761.56|              226.74|Sharable client-d...|        Jamieberg|   1|              Norway|2016-05-19 14:30:17|            0|
            |                   88.91| 33|   53852.85|              208.36|Enhanced dedicate...|      Brandonstad|   0|             Myanmar|2016-01-28 20:59:32|            0|
            |                    66.0| 48|   24593.33|              131.76|Reactive local ch...| Port Jefferybury|   1|           Australia|2016-03-07 01:40:15|            1|
            |                   74.53| 30|    68862.0|              221.51|Configurable cohe...|       West Colin|   1|             Grenada|2016-04-18 09:33:42|            0|
            |                   69.88| 20|   55642.32|              183.82|Mandatory homogen...|       Ramirezton|   1|               Ghana|2016-07-11 01:42:51|            0|
            |                   47.64| 49|   45632.51|              122.02|Centralized neutr...|  West Brandonton|   0|               Qatar|2016-03-16 20:19:01|            1|
            |                   83.07| 37|   62491.01|              230.87|Team-oriented gri...|East Theresashire|   1|             Burundi|2016-05-08 08:10:10|            0|
            |                   69.57| 48|   51636.92|              113.12|Centralized conte...|   West Katiefurt|   1|               Egypt|2016-06-03 01:14:41|            1|
            |                   79.52| 24|   51739.63|              214.23|Synergistic fresh...|       North Tara|   0|Bosnia and Herzeg...|2016-04-20 21:49:22|            0|
            |                   42.95| 33|    30976.0|              143.56|Grass-roots coher...|     West William|   0|            Barbados|2016-03-24 09:31:49|            1|
            |                   63.45| 23|   52182.23|              140.64|Persistent demand...|   New Travistown|   1|               Spain|2016-03-09 03:41:30|            1|
            |                   55.39| 37|   23936.86|              129.41|Customizable mult...|   West Dylanberg|   0|Palestinian Terri...|2016-01-30 19:20:41|            1|
            |                   82.03| 41|   71511.08|              187.53|Intuitive dynamic...|      Pruittmouth|   0|         Afghanistan|2016-05-02 07:00:58|            0|
            |                    54.7| 36|   31087.54|              118.39|Grass-roots solut...|      Jessicastad|   1|British Indian Oc...|2016-02-13 07:53:55|            1|
            |                   74.58| 40|   23821.72|              135.51|Advanced 24/7 pro...|       Millertown|   1|  Russian Federation|2016-02-27 04:43:07|            1|
            +------------------------+---+-----------+--------------------+--------------------+-----------------+----+--------------------+-------------------+-------------+*/
            // Imprecion del Schema del DataFrame
            data.printSchema()

            /*root
            |-- Daily Time Spent on Site: double (nullable = true)
            |-- Age: integer (nullable = true)
            |-- Area Income: double (nullable = true)
            |-- Daily Internet Usage: double (nullable = true)
            |-- Ad Topic Line: string (nullable = true)
            |-- City: string (nullable = true)
            |-- Male: integer (nullable = true)
            |-- Country: string (nullable = true)
            |-- Timestamp: timestamp (nullable = true)
            |-- Clicked on Ad: integer (nullable = true)*/
            ///////////////////////
            /// Despliegue los datos /////
            /////////////////////

            // Impresion de un renglon de ejemplo 

            data.head(1)
            /*res50: Array[org.apache.spark.sql.Row] = Array([68.95,35,61833.9,256.09,Cloned 5thgeneration orchestration,Wrightburgh,0,Tunisia,2016-03-27 00:53:11.0,0])
            */

            // creamos la variable colnames la cual contendra en un arreglo de string la informacion de la primera columna.
            val colnames = data.columns
            /*colnames: Array[String] = Array(Daily Time Spent on Site, Age, Area Income, Daily Internet Usage, Ad Topic Line, City, Male, Country, Timestamp, Clicked on Ad)*/

            //creacion de la variable fristrow la cual contendra el contendido de la primera columna de datos.
            val firstrow = data.head(1)(0)
            /*firstrow: org.apache.spark.sql.Row = [68.95,35,61833.9,256.09,Cloned 5thgeneration orchestration,Wrightburgh,0,Tunisia,2016-03-27 00:53:11.0,0]
            */
            println("\n")
            println("Example data row")
            for(ind <- Range(1, colnames.length)){
                println(colnames(ind))
                println(firstrow(ind))
                println("\n")
            }

            /*Example data row
            Age
            35


            Area Income
            61833.9


            Daily Internet Usage
            256.09


            Ad Topic Line
            Cloned 5thgeneration orchestration


            City
            Wrightburgh


            Male
            0


            Country
            Tunisia


            Timestamp
            2016-03-27 00:53:11.0


            Clicked on Ad
            0
            */

            ////////////////////////////////////////////////////
            //// Preparar el DataFrame para Machine Learning ////
            //////////////////////////////////////////////////

            //   Hacer lo siguiente:

            //    - Creamos una nueva clolumna llamada "Hour" del Timestamp conteniendo la  "Hour of the click"
            val timedata = data.withColumn("Hour",hour(data("Timestamp")))
            //    - Renombracion de la columna "Clicked on Ad" a "label"
            val logregdata = timedata.select(data("Clicked on Ad").as("label") ,$"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
            // Visualizacion del renombramiento de la columna Clicked on Ad a label.
            logregdata.show()
            /*+-----+------------------------+---+-----------+--------------------+----+----+
            |label|Daily Time Spent on Site|Age|Area Income|Daily Internet Usage|Hour|Male|
            +-----+------------------------+---+-----------+--------------------+----+----+
            |    0|                   68.95| 35|    61833.9|              256.09|   0|   0|
            |    0|                   80.23| 31|   68441.85|              193.77|   1|   1|
            |    0|                   69.47| 26|   59785.94|               236.5|  20|   0|
            |    0|                   74.15| 29|   54806.18|              245.89|   2|   1|
            |    0|                   68.37| 35|   73889.99|              225.58|   3|   0|
            |    0|                   59.99| 23|   59761.56|              226.74|  14|   1|
            |    0|                   88.91| 33|   53852.85|              208.36|  20|   0|
            |    1|                    66.0| 48|   24593.33|              131.76|   1|   1|
            |    0|                   74.53| 30|    68862.0|              221.51|   9|   1|
            |    0|                   69.88| 20|   55642.32|              183.82|   1|   1|
            |    1|                   47.64| 49|   45632.51|              122.02|  20|   0|
            |    0|                   83.07| 37|   62491.01|              230.87|   8|   1|
            |    1|                   69.57| 48|   51636.92|              113.12|   1|   1|
            |    0|                   79.52| 24|   51739.63|              214.23|  21|   0|
            |    1|                   42.95| 33|    30976.0|              143.56|   9|   0|
            |    1|                   63.45| 23|   52182.23|              140.64|   3|   1|
            |    1|                   55.39| 37|   23936.86|              129.41|  19|   0|
            |    0|                   82.03| 41|   71511.08|              187.53|   7|   0|
            |    1|                    54.7| 36|   31087.54|              118.39|   7|   1|
            |    1|                   74.58| 40|   23821.72|              135.51|   4|   1|
            +-----+------------------------+---+-----------+--------------------+----+----+*/

            // Importacion de las librerias y apis de VectorAssembler y Vectors
            import org.apache.spark.ml.feature.VectorAssembler
            import org.apache.spark.ml.linalg.Vectors
            // Creacion de un nuevo objecto VectorAssembler llamado assembler para los feature las cuales seran las columnas restantes del dataset como un arreglo "Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"
            val assembler = (new VectorAssembler()
                            .setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
                            .setOutputCol("features"))



            // Utilizamos randomSplit para crear datos de entrenamineto con 70% y de prueba 30% con los que estara interactuando nuestro algoritmo 
            val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)


            ///////////////////////////////
            // Configure un Pipeline ///////
            /////////////////////////////

            // Importacion de la libreria de Pipeline
            import org.apache.spark.ml.Pipeline
            // Creacion de un nuevo objeto de  LogisticRegression llamado lr
            val lr = new LogisticRegression()
            // Creacion de un nuevo  pipeline con los elementos: assembler la cual es nuestras factures, lr duentro objeto de LOgisticRegression
            val pipeline = new Pipeline().setStages(Array(assembler, lr))
            // creacion de la variable model la cual contendra el elemento de pipeline el cual contendra un ajuste (fit)  para el conjunto que nos encontramos entrenando

            val model = pipeline.fit(training)

            // Resultado del modelo con la trasformacion de los datos de prueba.

            val results = model.transform(test)
            results.show()
            /*+-----+------------------------+---+-----------+--------------------+----+----+--------------------+--------------------+--------------------+----------+
            |label|Daily Time Spent on Site|Age|Area Income|Daily Internet Usage|Hour|Male|            features|       rawPrediction|         probability|prediction|
            +-----+------------------------+---+-----------+--------------------+----+----+--------------------+--------------------+--------------------+----------+
            |    0|                   55.55| 19|   41920.79|              187.95|   2|   0|[55.55,19.0,41920...|[-2.7102846397563...|[0.06236920377658...|       1.0|
            |    0|                   56.39| 27|    38817.4|              248.12|  21|   1|[56.39,27.0,38817...|[0.05463065996793...|[0.51365426922010...|       0.0|
            |    0|                   57.11| 22|   59677.64|              207.17|  23|   1|[57.11,22.0,59677...|[1.30970615429441...|[0.78746398070386...|       0.0|
            |    0|                   58.18| 25|   69112.84|              176.28|   6|   1|[58.18,25.0,69112...|[-0.0246843144535...|[0.49382923471195...|       1.0|
            |    0|                   60.75| 42|   69775.75|              247.05|  13|   1|[60.75,42.0,69775...|[2.47843548677271...|[0.92261617237664...|       0.0|
            |    0|                   60.83| 19|   40478.83|              185.46|  20|   1|[60.83,19.0,40478...|[-1.5886140007782...|[0.16957898667744...|       1.0|
            |    0|                   61.72| 26|   67279.06|              218.49|   7|   0|[61.72,26.0,67279...|[2.87135936051353...|[0.94641233121297...|       0.0|
            |    0|                   62.26| 37|   77988.71|              166.19|  15|   0|[62.26,37.0,77988...|[-0.4392350769379...|[0.39192325019289...|       1.0|
            |    0|                   63.43| 29|   66504.16|              236.75|  10|   1|[63.43,29.0,66504...|[3.89286789923692...|[0.98002052266040...|       0.0|
            |    0|                   65.56| 25|   69646.35|              181.25|  12|   1|[65.56,25.0,69646...|[1.89972451867876...|[0.86986034341242...|       0.0|
            |    0|                   65.82| 39|    76435.3|              221.94|   5|   0|[65.82,39.0,76435...|[2.99744124473508...|[0.95245839685127...|       0.0|
            |    0|                   66.03| 22|   59422.47|              217.37|  23|   0|[66.03,22.0,59422...|[3.55019479000523...|[0.97208271289611...|       0.0|
            |    0|                   66.67| 33|   72707.87|              228.03|  20|   1|[66.67,33.0,72707...|[4.39979112205852...|[0.98786906212386...|       0.0|
            |    0|                   66.83| 46|   77871.75|              196.17|  23|   1|[66.83,46.0,77871...|[1.11309067843160...|[0.75270486032505...|       0.0|
            |    0|                   67.47| 24|   60514.05|              225.05|   7|   1|[67.47,24.0,60514...|[3.82312436062652...|[0.97860821409950...|       0.0|
            |    0|                   68.25| 29|    70324.8|              220.08|  16|   0|[68.25,29.0,70324...|[4.35632733866736...|[0.98733700342917...|       0.0|
            |    0|                   68.37| 35|   73889.99|              225.58|   3|   0|[68.37,35.0,73889...|[3.95372136540473...|[0.98117788685217...|       0.0|
            |    0|                   68.47| 28|   67033.34|              226.64|   0|   0|[68.47,28.0,67033...|[4.14732058962249...|[0.98443925180759...|       0.0|
            |    0|                   68.72| 27|   66861.67|              225.97|  17|   0|[68.72,27.0,66861...|[4.68571177796716...|[0.99085817866136...|       0.0|
            |    0|                   68.88| 37|    78119.5|              179.58|  19|   0|[68.88,37.0,78119...|[1.77248369212999...|[0.85476627058034...|       0.0|
            +-----+------------------------+---+-----------+--------------------+----+----+--------------------+--------------------+--------------------+----------+
            only showing top 20 rows
            */

            ////////////////////////////////////
            //// Evaluacion del modelo /////////////
            //////////////////////////////////

            // Para la utilizacion de Metrics y Evaluation importamos la libreria de  MulticlassMetrics
            import org.apache.spark.mllib.evaluation.MulticlassMetrics

            // Convertimos los resutalos de la prueba (test) en RDD utilizando .as y .rdd

            val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
            // Inicializamos el objeto MulticlassMetrics con el parametro predictionAndLabels.

            val metrics = new MulticlassMetrics(predictionAndLabels)
            // Impresion de la matrix de confucion
            println("Confusion matrix:")
            println(metrics.confusionMatrix)
            /*Confusion matrix:
            146.0  7.0    
            1.0    161.0 */

            // Imprimimos la exactitud de nuestra predicion la cual es de 0.97 la cual es muy buena.
            metrics.accuracy

            /*res58: Double = 0.9746031746031746    */