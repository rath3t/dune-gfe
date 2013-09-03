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

using namespace Dune;

template <int blocksize>
void compareMatrices(const Matrix<FieldMatrix<double,blocksize,blocksize> >& fdMatrix,
                     const Matrix<FieldMatrix<double,blocksize,blocksize> >& adolcMatrix)
{
  assert(fdMatrix.N() == adolcMatrix.N());
  assert(fdMatrix.M() == adolcMatrix.M());

  Matrix<FieldMatrix<double,blocksize,blocksize> > diff = fdMatrix;
  diff -= adolcMatrix;

  if ( (diff.frobenius_norm() / fdMatrix.frobenius_norm()) > 1e-4) {
    std::cout << "Matrices differ:" << std::endl;
    std::cout << "finite differences:" << std::endl;
    printmatrix(std::cout, fdMatrix, "finite differences", "--");
    std::cout << "ADOL-C:" << std::endl;
    printmatrix(std::cout, adolcMatrix, "ADOL-C", "--");
    assert(false);
  }
}


int testHarmonicEnergy() {

  const int dim = 1;
  typedef YaspGrid<dim> GridType;
  FieldVector<double,dim> l(1);
  std::array<int,dim> elements;
  std::fill(elements.begin(), elements.end(), 1);
  GridType grid(l,elements);

  typedef Q1LocalFiniteElement<double,double,dim> LocalFE;
  LocalFE localFiniteElement;

  size_t nDofs = localFiniteElement.localBasis().size();

  typedef UnitVector<double,3> TargetSpace;
  std::vector<TargetSpace> localSolution(nDofs);

  for (size_t i=0; i<nDofs; i++)
    localSolution[i] = FieldVector<double,3>(1.0);

  for (size_t i=0; i<localSolution.size(); i++)
    std::cout << localSolution[i] << std::endl;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  typedef Q1NodalBasis<GridType::LeafGridView,double> Q1Basis;
  Q1Basis q1Basis(grid.leafView());

  HarmonicEnergyLocalStiffness<GridType::LeafGridView,
                               Q1Basis::LocalFiniteElement,
                               TargetSpace> harmonicEnergyLocalStiffness;

    // Assembler using ADOL-C
  typedef TargetSpace::rebind<adouble>::other ATargetSpace;
  HarmonicEnergyLocalStiffness<GridType::LeafGridView,
                                Q1Basis::LocalFiniteElement,
                                ATargetSpace> harmonicEnergyADOLCLocalStiffness;
  LocalGeodesicFEADOLCStiffness<GridType::LeafGridView,
                                Q1Basis::LocalFiniteElement,
                                TargetSpace> localGFEADOLCStiffness(&harmonicEnergyADOLCLocalStiffness);


  harmonicEnergyLocalStiffness.assembleHessian(*grid.leafbegin<0>(),localFiniteElement, localSolution);

  localGFEADOLCStiffness.assembleHessian(*grid.leafbegin<0>(),localFiniteElement, localSolution);

  compareMatrices(harmonicEnergyLocalStiffness.A_, localGFEADOLCStiffness.A_);

    return 0;
}

int testCosseratEnergy() {

  size_t nDofs = 4;

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

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    typedef Q1NodalBasis<GridType::LeafGridView,double> Q1Basis;
    Q1Basis q1Basis(grid.leafView());

    ParameterTree parameterSet;
    ParameterTreeParser::readINITree("../cosserat-continuum.parset", parameterSet);

    const ParameterTree& materialParameters = parameterSet.sub("materialParameters");
    std::cout << "Material parameters:" << std::endl;
    materialParameters.report();


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


    cosseratEnergyLocalStiffness.assembleHessian(*grid.leafbegin<0>(),localFiniteElement, localSolution);

    localGFEADOLCStiffness.assembleHessian(*grid.leafbegin<0>(),localFiniteElement, localSolution);

    compareMatrices(cosseratEnergyLocalStiffness.A_, localGFEADOLCStiffness.A_);

    return 0;
}


int main()
{
   testHarmonicEnergy();
   testCosseratEnergy();
}
