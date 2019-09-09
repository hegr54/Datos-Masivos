def breaking_records(score:List[Int]): Unit={  //(String)
    var i:Int= 10
    println("Score: "+ i)
    //var score = List(10,5,20,20,4,5,2,25,1)
    var min = score(0)
    var minCount = 0;
    var max = score(0)
    var maxCount = 0;
    for(i <-score){
        println(" "+ i)
        if(i<min){
            min=i
            minCount+=1
            println("minimos: " + min)
        }
        else if(i > max){
            max=i
            maxCount+=1
            println("Maximos: " + max)
        }
    }
    println(score)
    //var lista=List(maxCount,minCount)
println("Resultados: " + "maximos: "+ maxCount + " " + "minimos: " + minCount)
}
var score = List(10,5,20,20,4,5,2,25,1)
/*var score = List(3,4,21,36,10,28,35,5,24,42)
breaking_records(score))
 //prueba 2*/
 breaking_records(score)



 ////arreglo ingresado desde codificacion
 def breaking_records(score:Array(String)){  //(String)
    var i:Int= 9
    println("Score: "+ i)
    var score = Array(10,5,20,20,4,5,2,25,1);
    //var score = Array(3,4,21,36,10,28,35,5,24,42);
    var minimos = score(0)
    var minCount = 0;
    var maximos = score(0)
    var maxCount = 0;
    for(i <-score){
        println(" "+ i)
        if(i<minimos){
            minimos=i
            minCount+=1
            println ("minimos: "+minimos)
        }
        else if(i > maximos){
            maximos=i
            maxCount+=1
            println("maximos: "+ maximos)
        }
    }
 println("Resultados: " + "maximos: "+ maxCount + " " + "minimos: " + minCount)



 