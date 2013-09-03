/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     speelpenning.cpp
 Revision: $Id: speelpenning.cpp 299 2012-03-21 16:08:40Z kulshres $
 Contents: speelpennings example, described in the manual

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include "config.h"

#include <adolc/adouble.h>            // use of active doubles
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
// gradient(.) and hessian(.)
#include <adolc/taping.h>             // use of taping

#include <iostream>
#include <vector>

#include <cstdlib>
#include <math.h>

namespace std
{
   adouble max(adouble a, adouble b) {
      return fmax(a,b);
   }

   adouble sqrt(adouble a) {
     return sqrt(a);
   }

   adouble abs(adouble a) {
     return fabs(a);
   }

   adouble pow(const adouble& a, const adouble& b) {
     return pow(a,b);
   }

   bool isnan(adouble a) {
     return std::isnan(a.value());
   }

   bool isinf(adouble a) {
     return std::isinf(a.value());
   }

}

#include <dune/grid/yaspgrid.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/localfunctions/lagrange/q1.hh>

#include <dune/gfe/realtuple.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/harmonicenergystiffness.hh>

using namespace Dune;


#if 1
template <class GridView, class LocalFiniteElement, class TargetSpace>
double
energy(const typename GridView::template Codim<0>::Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<TargetSpace>& localPureSolution)
{
    double pureEnergy;
    typedef RealTuple<adouble,1> ADTargetSpace;
    std::vector<ADTargetSpace> localSolution(localPureSolution.size());

    typedef typename TargetSpace::template rebind<adouble>::other ATargetSpace;

    HarmonicEnergyLocalStiffness<GridView,LocalFiniteElement,ATargetSpace> assembler;

    trace_on(1);

    adouble energy = 0;

    for (size_t i=0; i<localSolution.size(); i++)
      localSolution[i] <<= localPureSolution[i];

    energy = assembler.energy(element,localFiniteElement,localSolution);

    energy >>= pureEnergy;

    trace_off(1);

    return pureEnergy;
}
#endif

#if 0
template <class GridView, class LocalFiniteElement, class TargetSpace>
double
energy(const typename GridView::template Codim<0>::Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<TargetSpace>& localPureSolution)
{
    typedef RealTuple<adouble,1> ADTargetSpace;
    std::vector<ADTargetSpace> localSolution(localPureSolution.size());

    trace_on(1);

    for (size_t i=0; i<localSolution.size(); i++)
      localSolution[i] <<= localPureSolution[i];

    assert(element.type() == localFiniteElement.type());

    static const int gridDim = GridView::dimension;
    typedef typename GridView::template Codim<0>::Entity::Geometry Geometry;

    double pureEnergy;
    adouble energy = 0;
    LocalGeodesicFEFunction<gridDim, adouble, LocalFiniteElement, ADTargetSpace> localGeodesicFEFunction(localFiniteElement,
                                                                                                      localSolution);

    int quadOrder = (element.type().isSimplex()) ? (localFiniteElement.localBasis().order()-1) * 2
                                                 : localFiniteElement.localBasis().order() * 2 * gridDim;



    const Dune::QuadratureRule<double, gridDim>& quad
        = Dune::QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);

    for (size_t pt=0; pt<quad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();

        const double integrationElement = element.geometry().integrationElement(quadPos);

        const typename Geometry::JacobianInverseTransposed& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);

        double weight = quad[pt].weight() * integrationElement;

        // The derivative of the local function defined on the reference element
        Dune::FieldMatrix<adouble, TargetSpace::EmbeddedTangentVector::dimension, gridDim> referenceDerivative = localGeodesicFEFunction.evaluateDerivative(quadPos);

        // The derivative of the function defined on the actual element
        Dune::FieldMatrix<adouble, TargetSpace::EmbeddedTangentVector::dimension, gridDim> derivative(0);

        for (size_t comp=0; comp<referenceDerivative.N(); comp++)
            jacobianInverseTransposed.umv(referenceDerivative[comp], derivative[comp]);

        // Add the local energy density
        // The Frobenius norm is the correct norm here if the metric of TargetSpace is the identity.
        // (And if the metric of the domain space is the identity, which it always is here.)
        energy += weight * derivative.frobenius_norm2();

    }

    energy *= 0.5;
    energy >>= pureEnergy;

    trace_off(1);

    return pureEnergy;
}
#endif

/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {

  size_t n = 4;

  //std::cout << className< decltype(adouble() / double()) >() << std::endl;

  const int dim = 2;
  typedef YaspGrid<dim> GridType;
  FieldVector<double,dim> l(1);
  std::array<int,dim> elements = {{1, 1}};
  GridType grid(l,elements);

  typedef Q1LocalFiniteElement<double,double,dim> LocalFE;
  LocalFE localFiniteElement;

  typedef RealTuple<double,1> TargetSpace;
  std::vector<TargetSpace> localSolution(n);
  localSolution[0] = TargetSpace(0);
  localSolution[1] = TargetSpace(0);
  localSolution[2] = TargetSpace(1);
  localSolution[3] = TargetSpace(1);

  double laplaceEnergy = energy<GridType::LeafGridView,LocalFE, TargetSpace>(*grid.leafbegin<0>(), localFiniteElement, localSolution);

  std::cout << "Laplace energy: " << laplaceEnergy << std::endl;

  std::vector<double> xp(n);
  for (size_t i=0; i<n; i++)
    xp[i] = 1;

  double** H   = (double**) malloc(n*sizeof(double*));
  for(size_t i=0; i<n; i++)
      H[i] = (double*)malloc((i+1)*sizeof(double));
  hessian(1,n,xp.data(),H);                   // H equals (n-1)g since g is

  std::cout << "Hessian:" << std::endl;
  for(size_t i=0; i<n; i++) {
    for (size_t j=0; j<n; j++) {
      double value = (j<=i) ? H[i][j] : H[j][i];
      std::cout << value << "  ";
    }
    std::cout << std::endl;
  }

  // Get gradient
#if 0
    int n,i,j;
    size_t tape_stats[STAT_SIZE];

    cout << "SPEELPENNINGS PRODUCT (ADOL-C Documented Example)\n\n";
    cout << "number of independent variables = ?  \n";
    cin >> n;

    std::vector<double> xp(n);
    double  yp = 0.0;
    std::vector<adouble> x(n);
    adouble  y = 1;

    for(i=0; i<n; i++)
        xp[i] = (i+1.0)/(2.0+i);           // some initialization

    trace_on(1);                         // tag = 1, keep = 0 by default
    for(i=0; i<n; i++) {
        x[i] <<= xp[i];                  // or  x <<= xp outside the loop
        y *= x[i];
    } // end for
    y >>= yp;
    trace_off(1);

    tapestats(1,tape_stats);             // reading of tape statistics
    cout<<"maxlive "<<tape_stats[NUM_MAX_LIVES]<<"\n";
    // ..... print other tape stats

    double* g = new double[n];
    gradient(1,n,xp.data(),g);                  // gradient evaluation

    double** H   = (double**) malloc(n*sizeof(double*));
    for(i=0; i<n; i++)
        H[i] = (double*)malloc((i+1)*sizeof(double));
    hessian(1,n,xp.data(),H);                   // H equals (n-1)g since g is




    double errg = 0;                     // homogeneous of degree n-1.
    double errh = 0;
    for(i=0; i<n; i++)
        errg += fabs(g[i]-yp/xp[i]);       // vanishes analytically.
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            if (i>j)                         // lower half of hessian
                errh += fabs(H[i][j]-g[i]/xp[j]);
        } // end for
    } // end for
    cout << yp-1/(1.0+n) << " error in function \n";
    cout << errg <<" error in gradient \n";
    cout << errh <<" consistency check \n";
#endif
    return 0;
} // end main

