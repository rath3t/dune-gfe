#ifndef DUNE_GFE_TWISTEDSTRIP_GRIDFUNCTION_HH
#define DUNE_GFE_TWISTEDSTRIP_GRIDFUNCTION_HH

#include <cmath>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/curvedgrid/gridfunctions/analyticgridfunction.hh>

namespace Dune {

  /// \brief Functor representing a twisted strip in 3D, with length length_ and nTwists twists
  template <class T = double>
  class TwistedStripProjection
  {
    static const int dim = 3;
    using Domain = FieldVector<T,dim>;
    using Jacobian = FieldMatrix<T,dim,dim>;

    T length_;
    double nTwists_;

  public:
    TwistedStripProjection (T length, double nTwists)
      : length_(length),
      nTwists_(nTwists)
    {}

    /// \brief Project the coordinate to the twisted strip using closest point projection
    Domain operator() (const Domain& x) const
    {
      // Angle for the x - position of the point
      double alpha = std::acos(-1)*nTwists_*x[0]/length_;
      double cosalpha = std::cos(alpha);
      double sinalpha = std::sin(alpha);

      // Add the line from middle to the new point - the direction is (0,cosalpha,sinalpha)
      // We need the distance, calculated by (y^2 + z^2)^0.5
      double dist = std::sqrt(x[1]*x[1] + x[2]*x[2]);
      double y = std::abs(cosalpha*dist);
      if (x[1] < 0 and std::abs(x[1]) > std::abs(x[2])) { // only trust the sign of x[1] if it is larger than x[2]
        y *= (-1);
      } else if (std::abs(x[1]) <= std::abs(x[2])) {
        if (cosalpha * sinalpha >= 0 and x[2] < 0) // if cosalpha*sinalpha > 0, x[1] and x[2] must have the same sign
          y *= (-1);
        if (cosalpha * sinalpha <= 0 and x[2] > 0) // if cosalpha*sinalpha < 0, x[1] and x[2] must have opposite signs
          y *= (-1);
      }
      double z = std::abs(sinalpha*dist);
      if (x[2] < 0 and std::abs(x[2]) > std::abs(x[1])) {
        z *= (-1);
      } else if (std::abs(x[2]) <= std::abs(x[1])) {
        if (cosalpha * sinalpha >= 0 and x[1] < 0)
          z *= (-1);
        if (cosalpha * sinalpha <= 0 and x[1] > 0)
          z *= (-1);
      }
      return {x[0], y , z};
    }

    /// \brief derivative of the projection to the twistedstrip
    friend auto derivative (const TwistedStripProjection& twistedStrip)
    {
      DUNE_THROW(NotImplemented,"The derivative of the projection to the twisted strip is not implemented yet!");
      return [length = twistedStrip.length_, nTwists = twistedStrip.nTwists_](const Domain& x)
             {
               Jacobian out;
               return out;
             };
    }

    /// \brief Normal Vector of the twisted strip
    Domain normal (const Domain& x) const
    {
      // Angle for the x - position of the point
      double alpha = std::acos(-1)*nTwists_*x[0]/length_;
      std::cout << "normal at " << x[0] << "is " << -std::sin(alpha) << ", " << std::cos(alpha) << std::endl;
      return {0,-std::sin(alpha), std::cos(alpha)};
    }

    /// \brief The mean curvature of the twisted strip
    T mean_curvature (const Domain& /*x*/) const
    {
      DUNE_THROW(NotImplemented,"The mean curvature of the twisted strip is not implemented yet!");
      return 1;
    }

    /// \brief The area of the twisted strip
    T area (T width) const
    {
      return length_*width;
    }
  };

  /// \brief construct a grid function representing the parametrization of a twisted strip
  template <class Grid, class T>
  auto twistedStripGridFunction (T length, double nTwists)
  {
    return analyticGridFunction<Grid>(TwistedStripProjection<T>{length, nTwists});
  }

} // end namespace Dune

#endif // DUNE_GFE_TWISTEDSTRIP_GRIDFUNCTION_HH
