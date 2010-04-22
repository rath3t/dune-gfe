#ifndef GEODESIC_DIFFERENCE_HH
#define GEODESIC_DIFFERENCE_HH

#include <vector>

#include <dune/istl/bvector.hh>

template <class TargetSpace>
Dune::BlockVector<typename TargetSpace::TangentVector> computeGeodesicDifference(const std::vector<TargetSpace>& a,
                                                                                 const std::vector<TargetSpace>& b)
{
    if (a.size() != b.size())
        DUNE_THROW(Dune::Exception, "a and b have to have the same length!");

    Dune::BlockVector<typename TargetSpace::TangentVector> result(a.size());

    for (size_t i=0; i<result.size(); i++) {

        // Subtract orientations on the tangent space of 'a'
        result[i] = TargetSpace::difference(a[i], b[i]);

    }

    return result;
}

#endif
