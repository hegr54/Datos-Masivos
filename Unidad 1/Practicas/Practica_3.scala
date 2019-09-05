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
    for( k <- Range(0,num3)){
         b = b + a
         a = b - a
    }
    return a
}