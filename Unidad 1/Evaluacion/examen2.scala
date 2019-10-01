///1
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
///2
val df = spark.read.option("header", "true").option("inferSchema","true")csv("/home/rigoberto/Datos Masivos/Datos-Masivos/Unidad 1/Practicas/spark_dataframes/Netflix_2011_2016.csv")
///3

///4
df.printSchema()
///5
df.select($"Date",$"Open",$"High",$"Low",$"Close").show(5)
///6
df.describe().show()
///7
val newColumn = df.withColumn($"HV Ratio",df($"High")/df($"Volume"))
newColumn.select("HV Ratio").show()
///8
df.orderBy($"High".desc).show()
///9
println "tiene los valores de cuando cerro la bolsa de valor en netgflix"
///10
df.select(max($"Volume")).show()
df.select(min($"Volume")).show()
///11
df.filter($"Close"<600).count()
///12
(df.filter($"High">500).count() * 1.0 / df.count()) * 100
///13
df.select(corr($"High",$"Volume")).show()
///14
val dfyears = df.withColumn("Year",year(df("Date"))) ///creacion de la columna year 
val maximoyear = dfyears.select($"Year", $"High").groupBy("Year").max() ///seleccion de las columnas year y high, la creacion de grupos
val resultado = maximoyear.select($"Year", $"max(High)") ///los valores mayores de las dos columnas year y high
res.show() ///resultado
///15
val df2mes = df.withColumn("Month",month(df("Date"))) //creacion de la columna mes
val avgmes = df2mes.select($"Month",$"Close").groupBy("Month").mean() ///optenemos tres columnas que es el mes y con avg sacamos el promedio de cada fila
avgmes.select($"Month",$"avg(Close)").orderBy("Month").show() ///obtenemos los meses del year junto con el promedio de cierre de la bolsa de valor
