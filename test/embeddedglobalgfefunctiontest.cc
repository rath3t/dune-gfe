// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include "config.h"

#include <dune/grid/yaspgrid.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/gridfunctions/gridviewfunction.hh>

#include <dune/gfe/functions/embeddedglobalgfefunction.hh>
#include <dune/gfe/functions/localgeodesicfefunction.hh>
#include <dune/gfe/functions/localprojectedfefunction.hh>
#include <dune/gfe/spaces/unitvector.hh>

using namespace Dune;

int main(int argc, char** argv)
{
  MPIHelper::instance(argc, argv);

  // Make a test grid
  const int dim = 2;
  YaspGrid<dim> grid({1,1}, {5,5});

  auto gridView = grid.leafGridView();

  // Make a test basis
  using namespace Functions::BasisFactory;
  auto basis = Functions::BasisFactory::makeBasis(gridView, lagrange<2>());

  // Make a test coefficient set
  using TargetSpace = GFE::UnitVector<double,3>;

  std::vector<TargetSpace> coefficients(basis.size());
  std::fill(coefficients.begin(), coefficients.end(), FieldVector<double,3>({1,0,0}));

  using GeodesicInterpolationRule = GFE::LocalGeodesicFEFunction<dim, double, decltype(basis)::LocalView::Tree::FiniteElement, TargetSpace>;
  GFE::EmbeddedGlobalGFEFunction<decltype(basis),GeodesicInterpolationRule,TargetSpace> testFunction(basis, coefficients);

  // Evaluate the function at the element centers
  auto localGFEFunction = localFunction(testFunction);
  for (auto&& element : elements(gridView))
  {
    localGFEFunction.bind(element);
    std::cout << localGFEFunction({0.5, 0.5}) << std::endl;
  }

  // Can we use EmbeddedGlobalGFEFunction within the type erasure wrapper?
  auto typeErasure = Functions::makeGridViewFunction(testFunction, gridView);
};
