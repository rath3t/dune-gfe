#ifndef DUNE_GFE_MOEBIUSSTRIP_GRIDFUNCTION_HH
#define DUNE_GFE_MOEBIUSSTRIP_GRIDFUNCTION_HH

#include <cmath>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/curvedgrid/gridfunctions/analyticgridfunction.hh>

namespace Dune::GFE
{

  /// \brief Functor representing a Möbius strip in 3D, where the base circle is the circle in the x-y-plane with radius_ around 0,0,0
  template <class T = double>
  class MoebiusStripProjection
  {
    static const int dim = 3;
    using Domain = FieldVector<T,dim>;
    using Jacobian = FieldMatrix<T,dim,dim>;

    T radius_;

  public:
    MoebiusStripProjection (T radius)
      : radius_(radius)
    {}

    /// \brief Project the coordinate to the MS using closest point projection
    Domain operator() (const Domain& x) const
    {
      double nrm = std::sqrt(x[0]*x[0] + x[1]*x[1]);
      //center for the point x - it lies on the circle around (0,0,0) with radius radius_ in the x-y-plane
      Domain center{x[0] * radius_/nrm, x[1] * radius_/nrm, 0};
      double cosu = x[0]/nrm;
      double sinu = x[1]/nrm;
      double cosuhalf = std::sqrt((1+cosu)/2);
      cosuhalf *= (sinu < 0) ? (-1) : 1 ; // u goes from 0 to 2*pi, so cosuhalf is negative if sinu < 0, we multiply that here
      double sinuhalf = std::sqrt((1-cosu)/2); //u goes from 0 to 2*pi, so sinuhalf is always >= 0

      // now calculate line from center to the new point
      // the direction is (cosuhalf*cosu,cosuhalf*sinu,sinuhalf), as can be seen from the formula to construct the MS, we normalize this vector
      Domain centerToPointOnMS{cosuhalf*cosu,cosuhalf*sinu,sinuhalf};
      centerToPointOnMS /= centerToPointOnMS.two_norm();

      Domain centerToX = center - x;
      // We need the length, let theta be the angle between the vector centerToX and center to centerToPointOnMS
      // then cos(theta) = centerToX*centerToPointOnMS/(len(centerToX)*len(centerToPointOnMS))
      // We want to project the point to MS, s.t. the angle between the MS and the projection is 90deg
      // Then, length = cos(theta) * len(centerToX) = centerToX*centerToPointOnMS/len(centerToPointOnMS) = centerToX*centerToPointOnMS
      double length = - centerToX * centerToPointOnMS;
      centerToPointOnMS *= length;
      return center + centerToPointOnMS;
    }


    /// \brief derivative of the projection to the MS
    friend auto derivative (const MoebiusStripProjection& moebiusStrip)
    {
      DUNE_THROW(NotImplemented,"The derivative of the projection to the Möbius strip is not implemented yet!");
      return [radius = moebiusStrip.radius_](const Domain& x)
             {
               Jacobian out;
               return out;
             };
    }

    /// \brief Normal Vector of the MS
    Domain normal (const Domain& x) const
    {
      using std::sqrt;
      Domain nVec = {x[0],x[1],0};
      double nrm = std::sqrt(x[0]*x[0] + x[1]*x[1]);
      double cosu = x[0]/nrm;
      double sinu = x[1]/nrm;
      double cosuhalf = std::sqrt((1+cosu)/2);
      cosuhalf *= (sinu < 0) ? (-1) : 1 ; // u goes from 0 to 2*pi, so cosuhalf is negative if sinu < 0, we multiply that here
      double sinuhalf = std::sqrt((1-cosu)/2);
      nVec[2] = (-1)*cosuhalf*(cosu*x[0]+sinu*x[1])/sinuhalf;
      nVec /= nVec.two_norm();
      return nVec;
    }

    /// \brief The mean curvature of the MS
    T mean_curvature (const Domain& /*x*/) const
    {
      DUNE_THROW(NotImplemented,"The mean curvature of the Möbius strip is not implemented yet!");
      return 1;
    }

    /// \brief The area of the MS
    T area () const
    {
      DUNE_THROW(NotImplemented,"The area of the Möbius strip is not implemented yet!");
      return 1;
    }
  };

  /// \brief construct a grid function representing the parametrization of a MS
  template <class Grid, class T>
  auto moebiusStripGridFunction (T radius)
  {
    return analyticGridFunction<Grid>(MoebiusStripProjection<T>{radius});
  }

}  // namespace Dune::GFE

#endif // DUNE_GFE_MOEBIUSSTRIP_GRIDFUNCTION_HH
