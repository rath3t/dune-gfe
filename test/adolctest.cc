#include "config.h"

#include <adolc/adouble.h>            // use of active doubles
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
// gradient(.) and hessian(.)
#include <adolc/taping.h>             // use of taping

#undef overwrite  // stupid: ADOL-C sets this to 1, so the name cannot be used

#include <iostream>
#include <vector>

#include <cstdlib>
#include <math.h>

#include <dune/gfe/adolcnamespaceinjections.hh>

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>

#include <dune/grid/yaspgrid.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/localfunctions/lagrange/q1.hh>

#include <dune/gfe/realtuple.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/harmonicenergystiffness.hh>
#include <dune/gfe/cosseratenergystiffness.hh>

using namespace Dune;


#if 1
template <class GridView, class LocalFiniteElement, class TargetSpace>
double
energy(const typename GridView::template Codim<0>::Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<TargetSpace>& localPureSolution)
{
    double pureEnergy;

    typedef typename TargetSpace::template rebind<adouble>::other ATargetSpace;
    std::vector<ATargetSpace> localSolution(localPureSolution.size());

#if 0
    HarmonicEnergyLocalStiffness<GridView,LocalFiniteElement,ATargetSpace> assembler;
#else
    ParameterTree parameterSet;
    ParameterTreeParser::readINITree("../cosserat-continuum.parset", parameterSet);

    const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
    std::cout << "Material parameters:" << std::endl;
    materialParameters.report();

    CosseratEnergyLocalStiffness<GridView,LocalFiniteElement,3, adouble> assembler(materialParameters, NULL, NULL);
#endif

    trace_on(1);

    adouble energy = 0;

    for (size_t i=0; i<localSolution.size(); i++)
      localSolution[i] <<=localPureSolution[i];

    energy = assembler.energy(element,localFiniteElement,localSolution);

    energy >>= pureEnergy;

    trace_off(1);

    size_t tape_stats[STAT_SIZE];
    tapestats(1,tape_stats);             // reading of tape statistics
    cout<<"maxlive "<<tape_stats[NUM_MAX_LIVES]<<"\n";
    // ..... print other tape stats

    return pureEnergy;
}
#endif


/****************************************************************************/
/*                                                             MAIN PROGRAM */
int main() {

  size_t nDofs = 4;

  //std::cout << className< decltype(adouble() / double()) >() << std::endl;

  const int dim = 2;
  typedef YaspGrid<dim> GridType;
  FieldVector<double,dim> l(1);
  std::array<int,dim> elements = {{1, 1}};
  GridType grid(l,elements);

  typedef Q1LocalFiniteElement<double,double,dim> LocalFE;
  LocalFE localFiniteElement;

  //typedef RealTuple<double,1> TargetSpace;
  typedef RigidBodyMotion<double,3> TargetSpace;
  std::vector<TargetSpace> localSolution(nDofs);
  FieldVector<double,7> identity(0);
  identity[6] = 1;

  for (auto vIt = grid.leafbegin<dim>(); vIt != grid.leafend<dim>(); ++vIt) {
    localSolution[grid.leafView().indexSet().index(*vIt)].r = 0;
    for (int i=0; i<dim; i++)
      localSolution[grid.leafView().indexSet().index(*vIt)].r[i] = 2*vIt->geometry().corner(0)[i];
    localSolution[grid.leafView().indexSet().index(*vIt)].q = Rotation<double,3>::identity();
  }

  for (size_t i=0; i<localSolution.size(); i++)
    std::cout << localSolution[i] << std::endl;

  double laplaceEnergy = energy<GridType::LeafGridView,LocalFE, TargetSpace>(*grid.leafbegin<0>(), localFiniteElement, localSolution);

  std::cout << "Laplace energy: " << laplaceEnergy << std::endl;

  size_t nDoubles = nDofs*sizeof(TargetSpace) / sizeof(double);
  std::vector<double> xp(nDoubles);
  int idx=0;
  for (size_t i=0; i<nDofs; i++)
    for (size_t j=0; j<sizeof(TargetSpace) / sizeof(double); j++)
        xp[idx++] = localSolution[i].globalCoordinates()[j];

  // Compute gradient
    double* g = new double[nDoubles];
    gradient(1,nDoubles,xp.data(),g);                  // gradient evaluation

  int idx2 = 0;
  for(size_t i=0; i<nDofs; i++) {
    for (size_t j=0; j<sizeof(TargetSpace) / sizeof(double); j++) {
      std::cout << g[idx2++] << "  ";
    }
    std::cout << std::endl;
  }


  //exit(0);

  // Compute Hessian
  double** H   = (double**) malloc(nDoubles*sizeof(double*));
  for(size_t i=0; i<nDoubles; i++)
      H[i] = (double*)malloc((i+1)*sizeof(double));
  hessian(1,nDoubles,xp.data(),H);                   // H equals (n-1)g since g is

  std::cout << "Hessian:" << std::endl;
  for(size_t i=0; i<nDoubles; i++) {
    for (size_t j=0; j<nDoubles; j++) {
      double value = (j<=i) ? H[i][j] : H[j][i];
      std::cout << value << "  ";
    }
    std::cout << std::endl;
    exit(0);
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

