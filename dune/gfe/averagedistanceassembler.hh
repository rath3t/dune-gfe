#ifndef AVERAGE_DISTANCE_ASSEMBLER_HH
#define AVERAGE_DISTANCE_ASSEMBLER_HH

#include <vector>

#include <dune/gfe/symmetricmatrix.hh>


namespace Dune::GFE
{

/** \tparam TargetSpace The manifold that we are mapping to */
template <class TargetSpace, class WeightType=double>
class AverageDistanceAssembler
{
  typedef typename TargetSpace::ctype ctype;
  static const int size         = TargetSpace::TangentVector::dimension;
  static const int embeddedSize = TargetSpace::EmbeddedTangentVector::dimension;

public:

  /** \brief Constructor with given coefficients \f$ v_i \f$ and weights \f$ w_i \f$
   */
  AverageDistanceAssembler(const std::vector<TargetSpace>& coefficients,
                           const std::vector<WeightType>& weights)
    : coefficients_(coefficients),
    weights_(weights)
  {}

  /** \brief Constructor with given coefficients \f$ v_i \f$ and weights \f$ w_i \f$
   *
   * The weights are given as a vector of length-1 FieldVectors instead of as a
   * vector of doubles.  The reason is that these weights are actually the values
   * of Lagrange shape functions, and the dune-localfunction interface returns
   * shape function values this way.
   */
  AverageDistanceAssembler(const std::vector<TargetSpace>& coefficients,
                           const std::vector<Dune::FieldVector<WeightType,1> >& weights)
    : coefficients_(coefficients),
    weights_(weights.size())
  {
    for (size_t i=0; i<weights.size(); i++)
      weights_[i] = weights[i][0];
  }

  ctype value(const TargetSpace& x) const {

    ctype result = 0;
    for (size_t i=0; i<coefficients_.size(); i++) {
      ctype dist = TargetSpace::distance(coefficients_[i], x);
      result += weights_[i]*dist*dist;
    }

    return result;
  }

  void assembleEmbeddedGradient(const TargetSpace& x,
                                typename TargetSpace::EmbeddedTangentVector& gradient) const
  {
    gradient = 0;
    for (size_t i=0; i<coefficients_.size(); i++)
      gradient.axpy(weights_[i],
                    TargetSpace::derivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i], x));
  }

  void assembleGradient(const TargetSpace& x,
                        typename TargetSpace::TangentVector& gradient) const
  {
    typename TargetSpace::EmbeddedTangentVector embeddedGradient;
    assembleEmbeddedGradient(x,embeddedGradient);

    Dune::FieldMatrix<ctype,size,embeddedSize> orthonormalFrame = x.orthonormalFrame();
    orthonormalFrame.mv(embeddedGradient,gradient);
  }

  void assembleEmbeddedHessian(const TargetSpace& x,
                               SymmetricMatrix<ctype,embeddedSize>& matrix) const
  {
    matrix = 0;
    for (size_t i=0; i<coefficients_.size(); i++)
      matrix.axpy(weights_[i], TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i], x));
  }

  // TODO Use a Symmetric matrix for the result
  void assembleHessian(const TargetSpace& x,
                       Dune::FieldMatrix<ctype,size,size>& matrix) const
  {
    SymmetricMatrix<ctype,embeddedSize> embeddedHessian;
    assembleEmbeddedHessian(x,embeddedHessian);

    Dune::FieldMatrix<ctype,size,embeddedSize> frame = x.orthonormalFrame();

    // this is frame * embeddedHessian * frame^T
    for (int i=0; i<size; i++)
      for (int j=0; j<size; j++)
        matrix[i][j] = embeddedHessian.energyScalarProduct(frame[i], frame[j]);
  }

  const std::vector<TargetSpace> coefficients_;

  std::vector<WeightType> weights_;

};

}  // namespace Dune::GFE

#endif
