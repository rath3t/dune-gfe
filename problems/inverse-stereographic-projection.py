# The inverse stereographic projection through the north pole, and its derivative
def f(x):
    normSquared = x[0]*x[0]+x[1]*x[1]
    return [2*x[0] / (normSquared+1), 2*x[1] / (normSquared+1), (normSquared-1)/ (normSquared+1)]
    #a = 20
    #normSquared = a*a*(x[0]*x[0]+x[1]*x[1])
    #return [a*2*x[0] / (normSquared+1), a*2*x[1] / (normSquared+1), (normSquared-1)/ (normSquared+1)]

def df(x):
    normSquared = x[0]*x[0]+x[1]*x[1]
    denominator = (1+normSquared)*(1+normSquared)

    return (( (2*(1+normSquared) - 4*x[0]*x[0]) / denominator, (-4*x[0]*x[1]) / denominator),
            ( (-4*x[1]*x[0]) / denominator, (2*(1+normSquared) - 4*x[1]*x[1]) / denominator),
            (4*x[0]/denominator,4*x[1]/denominator))

fdf = (f, df)
