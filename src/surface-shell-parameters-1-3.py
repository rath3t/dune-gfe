import math

class SurfaceShellParameters:
    def thickness(self, x):
        if (x[1] > 50.0):
            return 1.0
        else:
            return 3.0

    #Lame parameters for the surface shell material:
    #Parameters for E = 20.4 MPa, nu=0.335
    #mu_cosserat = 7.64e+6
    #lambda_cosserat = 1.55e+7
    def lame(self, x):
        if (x[1] > 50.0):
            return [7.64e+6,1.55e+7]
        else:
            return [7.64e+6,1.55e+7]
