import math

# Wiggly initial curve on the sphere in R^3 for a curve-shortening flow simulation
#
# Note: This method does not have to return unit vectors.  Any non-zero vectors
# will do; they are normalized after reading anyway.

#
#
# Kurve
#
#def f(x):
#    return [math.sin(0.5*math.pi*x[0]), math.cos(0.5*math.pi*x[0]), math.sin(10*math.pi*x[0])]

#
#
#Spirale
#
#def f(x):
#	a=0.05
#	n=3
#	phi=2*n*math.pi*x[0]
#	radius= a*math.sqrt(phi)
#	normsquared = radius*radius
#	return [(2*radius*math.cos(phi))/(normsquared +1), (2*radius*math.sin(phi)) / (normsquared +1), (1-normsquared) / (normsquared +1)]

# 
#
#Doppelspirale
#
#def f(x):
#	a=0.05
#	n=3
#	phi = 4*n*math.pi*math.sqrt((1-2*x[0])*(1-2*x[0]))
#	radius = a*math.sqrt(phi)
#	radiusBeiViertel = a*math.sqrt(2*n*math.pi)
#
#	if 0<= x[0]< 0.25:
#		line= radiusBeiViertel-x[0]+0.25
#		normsquared = line*line
#		return [2*line / (normsquared +1), 0, (1-normsquared) / (normsquared +1)]
#	elif 0.25<= x[0]< 0.5:
#		normsquared = radius*radius
#		return [(2*radius*math.cos(phi))/(normsquared +1), (2*radius*math.sin(phi)) / (normsquared +1), (1-normsquared) / (normsquared +1)]
#	elif 0.5<= x[0] < 0.75:
#		normsquared = radius*radius
#		return [-(2*radius*math.cos(phi))/(normsquared +1), -(2*radius*math.sin(phi)) / (normsquared +1), (1-normsquared) / (normsquared +1)]
#	else:
#		line=-radiusBeiViertel-x[0]+0.75
#		normsquared = line*line
#		return [2*line / (normsquared +1), 0, (1-normsquared) / (normsquared +1)]


#
#
# Kreis
#def f(x):
#	phi= 2*math.pi*x[0]
#	radius=0.5
#	normsquared = radius*radius
#	return [(2*radius*math.cos(phi))/(normsquared +1), (2*radius*math.sin(phi)) / (normsquared +1), (1-normsquared) / (normsquared +1)]

# Kreis (flach)
#def f(x):
#	phi= 2*math.pi*x[0]
#	radius=0.5
#	return [radius*math.cos(phi), radius*math.sin(phi)]

# Ellipse
#def f(x):
#	phi= 2*math.pi*x[0]
#	radius1=0.5
#	radius2=0.05
#	normsquared = radius1*radius1+(radius2*radius2*-radius1*radius1)*math.sin(phi)*math.sin(phi)
#	return [(2*radius1*math.cos(phi))/(normsquared +1), (2*radius2*math.sin(phi)) / (normsquared +1), (1-normsquared) / (normsquared +1)]


#
#geschlossene Doppelspirale 
#
def f(x):
	a=0.1
	n=2
	b=0.05
	z= (x[0]-2*b+0.5)*0.5*1/(1-2*b)
	phi1 = 4*n*math.pi*math.sqrt((1-2*z)*(1-2*z))
	phi2= (2*n+1)*math.pi*math.sqrt((1-2*x[0])*(1-2*x[0]))
	radius1 = a*math.sqrt(phi1)
	radius2 = a*math.sqrt(phi2)

	if 0<= x[0]< b:
		line= 4*a*(math.sqrt(2*n*math.pi)-math.sqrt((2*n+1)*math.pi))*x[0]+ a*math.sqrt((2*n+1)*math.pi)
		normsquared = line*line
		return [2*line / (normsquared +1), 0, (1-normsquared) / (normsquared +1)]
	elif b<= x[0]< 0.5:
		normsquared = radius1*radius1
		return [(2*radius1*math.cos(phi1))/(normsquared +1), (2*radius1*math.sin(phi1)) / (normsquared +1), (1-normsquared) / (normsquared +1)]
	else:
		normsquared = radius2*radius2
		return [-(2*radius2*math.cos(phi2))/(normsquared +1), -(2*radius2*math.sin(phi2)) / (normsquared +1), (1-normsquared) / (normsquared +1)]

