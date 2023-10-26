// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_GFE_WEIGHTED_SUM_ENERGY_HH
#define DUNE_GFE_WEIGHTED_SUM_ENERGY_HH

#include <dune/common/fmatrix.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/localgeodesicfefunction.hh>

template<class Basis, class TargetSpace>
class WeightedSumEnergy
  : public Dune::GFE::LocalEnergy<Basis,TargetSpace>
{
  // grid types
  typedef typename Basis::GridView GridView;
  typedef typename GridView::ctype DT;
  typedef typename TargetSpace::ctype RT;

  // some other sizes
  constexpr static int gridDim = GridView::dimension;

public:

  std::vector<std::shared_ptr<Dune::GFE::LocalEnergy<Basis,TargetSpace> > > addends_;

  std::vector<double> weights_;

  WeightedSumEnergy(std::vector<std::shared_ptr<Dune::GFE::LocalEnergy<Basis,TargetSpace> > > addends,
                    std::vector<double> weights)
  : addends_(addends),
    weights_(weights)
  {}

  /** \brief Assemble the energy for a single element */
  RT energy (const typename Basis::LocalView& localView,
             const std::vector<TargetSpace>& localConfiguration) const override

  {
    RT energy = 0;

    assert(weights_.size() == addends_.size());

    for (size_t i=0; i<addends_.size(); i++)
      energy += weights_[i] * addends_[i]->energy(localView, localConfiguration);

    return energy;
  }

};

#endif

