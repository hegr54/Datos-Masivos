import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Datos-Masivos/spark_dataframes/CitiGroup2006_2008")

df.printSchema()

import spark.implicits._

//funciones agregadas
///Practica 4 crear 20 funciones con esta base de datos
///a
df.select(min("Open")).show()       //Devuelve el valor mínimo de todos los valores numéricos no NULL especificados por la expresión, en el contexto del ámbito especificado. Puede usarla para especificar un valor mínimo para el eje del gráfico para controlar la escala.
df.select($"Open".as("OPEN")).show()    //Solo se le genera un alias como una etiqueta para facilitar algunos nombres de tablas o atributos.
df.select($"Low".name("LOW")).show()  //Cambia el valor de la tabla
df.select(count($"Low")).show()     //Devuelve un recuento de los valores no NULL especificados por la expresión, que se evalúa en el contexto del ámbito indicado.
df.select(avg($"Open")).show()      //Devuelve el promedio de todos los valores numéricos no NULL especificados por la expresión, que se evalúa en el contexto del ámbito especificado.
df.select(countDistinct($"Open")).show()        //Devuelve un recuento de todos los valores no NULL distintos especificados por la expresión, que se evalúa en el contexto del ámbito especificado.
df.select(stddev("Low")).show() ///Devuelve la desviación estándar de todos los valores numéricos no NULL especificados por la expresión, que se evalúa en el contexto del ámbito especificado.
df.select(corr("High", "Low")).show()       //Devuelve la correlacion
df.select(stddev_pop("Low")).show        //Devuelve la desviación estándar de población de todos los valores numéricos no NULL especificados por la expresión, que se evalúa en el contexto del ámbito especificado.
df.select(sum($"Low")).show()       //Devuelve la suma de todos los valores numéricos no NULL especificados por la expresión, que se evalúa en el contexto del ámbito especificado.
df.select(variance($"Open")).show()        //Devuelve la varianza de todos los valores numéricos no NULL especificados por la expresión, que se evalúa en el contexto del ámbito especificado.
df.select(var_pop($"Open")).show()        //Devuelve la varianza de población de todos los valores numéricos no NULL especificados por la expresión, que se evalúa en el contexto del ámbito especificado.
df.select(var_samp($"Open")).show()        //El resultado de la función VAR_SAMP es equivalente a la desviación cuadrada estándar de la muestra del mismo conjunto de valores.
df.select(countDistinct($"Open")).show()    //Devuelve un recuento de todos los valores distintos no nulos especificados por la expresión, evaluados en el contexto del ámbito dado.
df.select(sumDistinct($"Open")).show()      //Además de sumar un total, también puede sumar un conjunto distante de valores.
df.select(collect_set($"Open")).show()      //devuelve un conjunto de objetos con elementos duplicados eliminados.
df.select(collect_list($"Open")).show()     // devuelve una lista de objetos con duplicados.
df.select(max($"High")).show()              // devuelve el valor máximo de la expresión en un grupo.
df.select(log($"High")).show()      //Devuelve el primer logaritmo basado en argumentos del segundo argumento.
df.select(length($"Open")).show()       //Calcula la longitud de una cadena o columna binaria dada.
df.select(skewness($"Open")).show()     //devuelve el sesgo de los valores en un grupo.
