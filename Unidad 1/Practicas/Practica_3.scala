//susecion de fibonacci
//////Primera Susecion fibonacci

def fib1(num1:Int):Int={
    if(num1<2)
    {
        return  num1
    }
    else
    {
        return fib1(num1-1)+fib1(num1-2)
    }
}

/////Segunda Susecion Fibonacci

def fib2(num2:Int): Double = {
    if(num2<2){
        return num2
    }
    else
    {
        var x=((1+math.sqrt(5))/2)
        var y=((math.pow(x,num2)-math.pow((1-x),num2))/(math.sqrt(5)))
        return y
    }
}

/////Tercero Susecion Fibonacci

def fib3(num3:Int):Int = {
    var a=0
    var b=1
    var c=0
    for(c <- Range(0,num2)) {
        val c = b + a
        a=b
        b=c
    }
    return a
}

////Cuarta sucesion fibonacci

def fib4(num4:Int):Int={
    var a=0
    var b=1
    for( k <- Range(0,num4)){
         b = b + a
         a = b - a
    }
    return a
}

////Quinta Sucesion fibonacci
//
def fib5(num5: Int): Int = {
    val f: Array[Int] = Array.ofDim[Int](num5 + 2)
    f(0) = 0
    f(1) = 1
    for (i <- Range (2, num5 + 1)) {
      
      f(i) = f(i - 1) + f(i - 2) //{ i += 1; i - 1 }
    }
    return f(num5)
  }

  ////Sexto sucecion Fibonacci

  def fib6 (n : Int) : Double =
{
    if (n <= 0)
    {
        return 0
    }
    var i = n-1
    var auxOne = 0.0
    var auxTwo = 1.0
    var ab = Array(auxTwo,auxOne)
    var cd = Array(auxOne,auxTwo)
    while (i>0)
    {
        if (i % 2 != 0)
        {
            auxOne = cd(1) * ab(1) + cd(0) * ab(0)
            auxTwo = cd(1) * (ab(1)+ab(0)) + cd(0)*ab(1)
            ab(0) = auxOne
            ab(1) = auxTwo 
        } 
        auxOne = (math.pow(cd(0),2)) + (math.pow(cd(1),2))
        auxTwo = cd(1)* (2*cd(0) + cd(1))
        cd(0) = auxOne
        cd(1) = auxTwo
        i = i/2
    }
    return (ab(0) + ab(1))
}
println(fib1(10))