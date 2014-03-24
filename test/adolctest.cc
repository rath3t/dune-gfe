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

#include <dune/istl/io.hh>

#include <dune/grid/yaspgrid.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/localfunctions/lagrange/q1.hh>

#include <dune/fufem/functionspacebases/q1nodalbasis.hh>

#include <dune/gfe/realtuple.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/harmonicenergystiffness.hh>
#include <dune/gfe/cosseratenergystiffness.hh>
#include <dune/gfe/localgeodesicfeadolcstiffness.hh>

#include "valuefactory.hh"
#include "multiindex.hh"

using namespace Dune;

/** \brief Computes the diameter of a set */
template <class TargetSpace>
double diameter(const std::vector<TargetSpace>& v)
{
    double d = 0;
    for (size_t i=0; i<v.size(); i++)
        for (size_t j=0; j<v.size(); j++)
            d = std::max(d, TargetSpace::distance(v[i],v[j]));
    return d;
}



template <int blocksize, class TargetSpace>
void compareMatrices(const Matrix<FieldMatrix<double,blocksize,blocksize> >& fdMatrix,
                     const Matrix<FieldMatrix<double,blocksize,blocksize> >& adolcMatrix,
                     const std::vector<TargetSpace>& configuration)
{
  assert(fdMatrix.N() == adolcMatrix.N());
  assert(fdMatrix.M() == adolcMatrix.M());

  Matrix<FieldMatrix<double,blocksize,blocksize> > diff = fdMatrix;
  diff -= adolcMatrix;

  if ( (diff.frobenius_norm() / fdMatrix.frobenius_norm()) > 1e-4) {

    std::cout << "Matrices differ for" << std::endl;
    for (size_t j=0; j<configuration.size(); j++)
      std::cout << configuration[j] << std::endl;

    std::cout << "finite differences:" << std::endl;
    printmatrix(std::cout, fdMatrix, "finite differences", "--");
    std::cout << "ADOL-C:" << std::endl;
    printmatrix(std::cout, adolcMatrix, "ADOL-C", "--");
    assert(false);
  }
}

template <class TargetSpace, int gridDim>
int testHarmonicEnergy() {

  std::cout << " --- Testing " << className<TargetSpace>() << ", domain dimension: " << gridDim << " ---" << std::endl;

  // set up a test grid
  typedef YaspGrid<gridDim> GridType;
  FieldVector<double,gridDim> l(1);
  std::array<int,gridDim> elements;
  std::fill(elements.begin(), elements.end(), 1);
  GridType grid(l,elements);

  typedef Q1NodalBasis<typename GridType::LeafGridView,double> Q1Basis;
  Q1Basis q1Basis(grid.leafGridView());

  typedef Q1LocalFiniteElement<double,double,gridDim> LocalFE;
  LocalFE localFiniteElement;

  // Assembler using finite differences
  HarmonicEnergyLocalStiffness<typename GridType::LeafGridView,
                               typename Q1Basis::LocalFiniteElement,
                               TargetSpace> harmonicEnergyLocalStiffness;

  // Assembler using ADOL-C
  typedef typename TargetSpace::template rebind<adouble>::other ATargetSpace;
  HarmonicEnergyLocalStiffness<typename GridType::LeafGridView,
                               typename Q1Basis::LocalFiniteElement,
                               ATargetSpace> harmonicEnergyADOLCLocalStiffness;
  LocalGeodesicFEADOLCStiffness<typename GridType::LeafGridView,
                                typename Q1Basis::LocalFiniteElement,
                                TargetSpace> localGFEADOLCStiffness(&harmonicEnergyADOLCLocalStiffness);

  size_t nDofs = localFiniteElement.localBasis().size();

  std::vector<TargetSpace> localSolution(nDofs);

  std::vector<TargetSpace> testPoints;
  ValueFactory<TargetSpace>::get(testPoints);

  int nTestPoints = testPoints.size();

  MultiIndex index(nDofs, nTestPoints);
  int numIndices = index.cycle();

  for (int i=0; i<numIndices; i++, ++index) {

    for (size_t j=0; j<nDofs; j++)
      localSolution[j] = testPoints[index[j]];

    if (diameter(localSolution) > 0.5*M_PI)
        continue;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::vector<typename TargetSpace::TangentVector> localGradient;
    harmonicEnergyLocalStiffness.assembleGradientAndHessian(*grid.template leafbegin<0>(),localFiniteElement, localSolution,
                                                            localGradient);

    localGFEADOLCStiffness.assembleGradientAndHessian(*grid.template leafbegin<0>(),localFiniteElement, localSolution,
                                                      localGradient);

    compareMatrices(harmonicEnergyLocalStiffness.A_, localGFEADOLCStiffness.A_, localSolution);

  }

  return 0;
}

int testCosseratEnergy() {

  typedef RigidBodyMotion<double,3> TargetSpace;
  const int gridDim = 2;

  std::cout << " --- Testing " << className<TargetSpace>() << ", domain dimension: " << gridDim << " ---" << std::endl;

  ParameterTree parameterSet;
  ParameterTreeParser::readINITree("../cosserat-continuum.parset", parameterSet);

  const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
  std::cout << "Material parameters:" << std::endl;
  materialParameters.report();

  // set up a test grid
  typedef YaspGrid<gridDim> GridType;
  FieldVector<double,gridDim> l(1);
  std::array<int,gridDim> elements;
  std::fill(elements.begin(), elements.end(), 1);
  GridType grid(l,elements);

  typedef Q1NodalBasis<typename GridType::LeafGridView,double> Q1Basis;
  Q1Basis q1Basis(grid.leafGridView());

  typedef Q1LocalFiniteElement<double,double,gridDim> LocalFE;
  LocalFE localFiniteElement;

  // Assembler using finite differences
  CosseratEnergyLocalStiffness<GridType::LeafGridView,
                               Q1Basis::LocalFiniteElement,
                               3> cosseratEnergyLocalStiffness(materialParameters,NULL,NULL);

  // Assembler using ADOL-C
  CosseratEnergyLocalStiffness<GridType::LeafGridView,
                               Q1Basis::LocalFiniteElement,
                               3,adouble> cosseratEnergyADOLCLocalStiffness(materialParameters, NULL, NULL);
  LocalGeodesicFEADOLCStiffness<GridType::LeafGridView,
                                Q1Basis::LocalFiniteElement,
                                TargetSpace> localGFEADOLCStiffness(&cosseratEnergyADOLCLocalStiffness);

  size_t nDofs = localFiniteElement.localBasis().size();

  std::vector<TargetSpace> localSolution(nDofs);
  std::vector<TargetSpace> testPoints;
  ValueFactory<TargetSpace>::get(testPoints);

  int nTestPoints = testPoints.size();

  MultiIndex index(nDofs, nTestPoints);
  int numIndices = index.cycle();

  for (int i=0; i<numIndices; i++, ++index) {

    for (size_t j=0; j<nDofs; j++)
      localSolution[j] = testPoints[index[j]];

    if (diameter(localSolution) > 0.5*TargetSpace::convexityRadius)
        continue;

    std::vector<typename TargetSpace::TangentVector> localGradient;
    cosseratEnergyLocalStiffness.assembleGradientAndHessian(*grid.leafbegin<0>(),localFiniteElement, localSolution,
                                                            localGradient);

    localGFEADOLCStiffness.assembleGradientAndHessian(*grid.leafbegin<0>(),localFiniteElement, localSolution,
                                                      localGradient);

    compareMatrices(cosseratEnergyLocalStiffness.A_, localGFEADOLCStiffness.A_, localSolution);

  }

  return 0;
}


int main()
{
   testHarmonicEnergy<UnitVector<double,2>, 1>();
   testHarmonicEnergy<UnitVector<double,3>, 1>();
   testHarmonicEnergy<UnitVector<double,2>, 2>();
   testHarmonicEnergy<UnitVector<double,3>, 2>();

   testCosseratEnergy();
}
