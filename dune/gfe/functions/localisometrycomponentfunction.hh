#ifndef DUNE_GFE_FUNCTIONS_LOCALISOMETRYCOMPONENTFUNCTION_HH
#define DUNE_GFE_FUNCTIONS_LOCALISOMETRYCOMPONENTFUNCTION_HH

#include <dune/gfe/bendingisometryhelper.hh>

namespace Dune::GFE::Impl
{
  /**
   * \brief Local isometry component function that can be used as a Dune::Functions::DifferentiableFunction
   *
   * \tparam ctype Type used for coordinates on the reference element
   * \tparam LocalInterpolation Local function mapping into a product manifold of type RealTuple x Rotations
   */
  template<class ctype, class LocalInterpolation>
  class LocalIsometryComponentFunction
  {
  public:
    LocalIsometryComponentFunction(LocalInterpolation& localInterpolation,
                                   const int component)
      : localInterpolation_(localInterpolation),
      component_(component)
    {}

    auto operator() (const Dune::FieldVector<ctype,2>& x) const
    {
      return localInterpolation_.evaluate(x)[Dune::Indices::_0].globalCoordinates()[component_];
    }

    /**
     * \brief Obtain derivative function
     */
    friend auto derivative(const LocalIsometryComponentFunction& p)
    {
      return [&p](const Dune::FieldVector<ctype,2>& x)
             {
               auto derivativeQuaternion = p.localInterpolation_.evaluate(x)[Dune::Indices::_1];
               auto derivativeMatrix = Rotation<ctype,3>::quaternionToMatrix(derivativeQuaternion);
               auto reducedDerivativeMatrix = cancelLastMatrixColumn(derivativeMatrix);
               return reducedDerivativeMatrix[p.component_];
             };
    }
    LocalInterpolation& localInterpolation_;
    const int component_;
  };

} // end namespace Dune::GFE::Impl

#endif   //#ifndef DUNE_GFE_FUNCTIONS_LOCALISOMETRYCOMPONENTFUNCTION_HH
