# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:28:35 2019

@author: narendra.b.sinappa
"""

# implementation of gradient descent.



import matplotlib.pyplot as plt
import numpy as np
#import sympy

# generating evenly spaced random numbers within the specified intervals and plotted using the function ex : ((x ** 3)-(3 *(x ** 2)))+7
function = lambda x:((x ** 3)-(3 *(x ** 2)))+7
x = np.linspace(-1,3,100)
plt.plot(x,function(x), color = 'r')
"""
# using sympy to find the first drivative of the given function
x = sympy.symbols('x')
fun = ((x ** 3)-(3 *(x ** 2)))+7
#first derivative
f = sympy.diff(fun,x)
"""
# Derivative of the function defined above
def Gradient(x):
    x_grad = ((3*(x**2)) - (6*x))
    return x_grad
"""
1) Function "Gradient" is to find the new value of x based on the derivative of a  function f(x).
2) finding derivative is to make roll the ball in negative direction, and find the local minima i.e y = small.
3)number of steps will be more if the learning rate is very low(ex : 0.000001).
4)while loop is used to find the local minima, Loop will execute until the condition fails. i.e ((new_x - prev_x) > precision) 
"""

def descend(new_x,prev_x,precision,Learning_rate):
    x_list,y_list = [new_x],[function(new_x)]
    while (abs(new_x-prev_x)>precision):
        prev_x = new_x
        x_grad = Gradient(prev_x)
        new_x = prev_x - (x_grad*Learning_rate)
        x_list.append(new_x)
        y_list.append(function(new_x))
    print(x_list,y_list)
    print(new_x)
    print(len(x_list))
    plt.scatter(x_list,y_list,color = 'g')
    plt.plot(x_list,y_list,color = 'r')
    plt.title("Gradient descent w.r.t f(x) = (x ** 3)-(3 *(x ** 2))+7")
    plt.xlabel("Evenly spaced points between -1 and 3------>")
    plt.ylabel("Descend of y---->")


#call the descend function, Values are randomly choosen.     
descend(0.5,0,0.0001,0.1)

        
        
    



