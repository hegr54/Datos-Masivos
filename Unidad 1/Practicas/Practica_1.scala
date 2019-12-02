//1.  algoritmo en scala que calcule el radio de un circulo
def Circulo(valor:Double)={
    var pi: Double=3.1416
    var Circunferencia: Double= valor
    var rest: Double=Circunferencia/(2*pi)
    println (s"El radio de un Circulo con un circunferencia de $Circunferencia es : "+ rest)
}
//2.algoritmo en scala que me diga si un numero es primo
def primo(x: Int) ={
    var n1: Double=x
    var residuo: Double= (n1%2)
    if (x%2==0){
         println (s"El numero es primo $n1 su residuo es $residuo")
    }
    else
    {
     println(s"no es primo el numero $n1 su residuo es $residuo")
     }
}
//3. Dada la variable bird = "tweet", utiliza interpolacion de string para
//   imprimir "Estoy ecribiendo un tweet"
var bird="tweet"
println(s"Estoy escribiendo un $bird")
//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la
//   secuencia "Luke"

var mensaje="Hola Luke yo soy tu padre!"
mensaje slice(5,10)

//5. Cual es la diferencia en value y una variable en scala?

 /* 
--- una variable "value" consiste en que es una variable inmutable ya que no se puede modificar sino hasta que se 
acabe el ciclo de ejecucion y no puedes resignar un valor.
--- las variables "variable" consiste en que es una variable mutable por lo que esta variable puede ser modificable
en cualquier momento.*/

//6. Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el numero 3.1416 

val tupla=(2,4,5,1,2,3,3.1416,23)
tupla._7