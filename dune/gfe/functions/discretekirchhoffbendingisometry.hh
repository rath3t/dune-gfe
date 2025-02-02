#ifndef DUNE_GFE_FUNCTIONS_DISCRETEKIRCHHOFFBENDINGISOMETRY_HH
#define DUNE_GFE_FUNCTIONS_DISCRETEKIRCHHOFFBENDINGISOMETRY_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>
#include <dune/gfe/functions/localisometrycomponentfunction.hh>
#include <dune/gfe/functions/localprojectedfefunction.hh>
#include <dune/gfe/bendingisometryhelper.hh>
#include <dune/gfe/tensor3.hh>

#include <dune/functions/functionspacebases/cubichermitebasis.hh>

namespace Dune::GFE
{
  /** \brief Interpolate a Discrete Kirchhoff finite element function from a set of RealTuple x Rotation coefficients
   *
   * The Discrete Kirchhoff finite element is a Hermite-type element.
   * The Rotation component of the coefficient set represents the deformation gradient.
   *
   * The class stores both a global coefficient vector 'coefficients_'
   * as well as a local coefficient vector 'localHermiteCoefficients_'
   * that is used for local evaluation after binding to an element.
   *
   *
   * \tparam DiscreteKirchhoffBasis The basis used to compute function values
   * \tparam CoefficientBasis The basis used to index the coefficient vector.
   * \tparam Coefficients The container of global coefficients
   */
  template <class DiscreteKirchhoffBasis, class CoefficientBasis, class Coefficients>
  class DiscreteKirchhoffBendingIsometry
  {
    using GridView = typename DiscreteKirchhoffBasis::GridView;
    using ctype = typename GridView::ctype;

  public:
    constexpr static int gridDim = GridView::dimension;
    using Coefficient = typename Coefficients::value_type;
    using RT = typename Coefficient::ctype;

    static constexpr int components_ = 3;

    typedef GFE::LocalProjectedFEFunction<CoefficientBasis, GFE::ProductManifold<RealTuple<RT,components_>, Rotation<RT,components_> > > LocalInterpolationRule;

    /** \brief The type used for derivatives */
    typedef Dune::FieldMatrix<RT, components_, gridDim> DerivativeType;

    /** \brief The type used for discrete Hessian */
    typedef Tensor3<RT, components_, gridDim, gridDim> discreteHessianType;


    /** \brief Constructor
     * \param basis An object of Hermite basis type
     * \param coefficientBasis An object of type LagrangeBasis<GridView,1>
     * \param globalIsometryCoefficients Values and derivatives of the function at the Lagrange points
     */
    DiscreteKirchhoffBendingIsometry(const DiscreteKirchhoffBasis& discreteKirchhoffBasis,
                                     const CoefficientBasis& coefficientBasis,
                                     Coefficients& globalIsometryCoefficients)
      : coefficientBasis_(coefficientBasis),
      basis_(discreteKirchhoffBasis),
      localView_(basis_),
      localViewCoefficient_(coefficientBasis_),
      globalIsometryCoefficients_(globalIsometryCoefficients)
    {}

    void bind(const typename GridView::template Codim<0>::Entity& element)
    {
      localView_.bind(element);

      /** extract the local coefficients from the global coefficient vector */
      std::vector<Coefficient> localIsometryCoefficients;
      getLocalIsometryCoefficients(localIsometryCoefficients,element);
      updateLocalHermiteCoefficients(localIsometryCoefficients,element);

      /**
       * @brief  Get Coefficients of the discrete Gradient
       *
       * The discrete Gradient is a special linear combination represented in a [P2]^2 - (Lagrange) space
       * The coefficients of this linear combination correspond to certain linear combinations of the Gradients of
       * the deformation (hermite) basis. The coefficients are stored in the form [Basisfunctions x components x gridDim]
       * in a BlockVector<FieldMatrix<RT, 3, gridDim>>.
       */
      Impl::discreteGradientCoefficients(discreteJacobianCoefficients_,rotationLFE_,*this,element);
    }

    /** bind to an element and update the coefficient member variables for a set of new local coefficients */
    void bind(const typename GridView::template Codim<0>::Entity& element,const Coefficients& newLocalCoefficients)
    {
      this->bind(element);

      // Update the global isometry coefficients.
      for (unsigned int vertex=0; vertex<3; ++vertex)
      {
        size_t localIdx = localViewCoefficient_.tree().localIndex(vertex);
        size_t globalIdx = localViewCoefficient_.index(localIdx);
        globalIsometryCoefficients_[globalIdx] = newLocalCoefficients[vertex];
      }

      // Update the local hermite coefficients.
      updateLocalHermiteCoefficients(newLocalCoefficients,element);

      // After setting a new local configurarion, the coefficients of the discrete Gradient need to be updated as well.
      Impl::discreteGradientCoefficients(discreteJacobianCoefficients_,rotationLFE_,*this,element);
    }

    /** \brief The type of the element we are bound to */
    Dune::GeometryType type() const
    {
      return localView_.element().type();
    }


    /** \brief Evaluate the function */
    auto evaluate(const Dune::FieldVector<ctype, gridDim>& local) const
    {
      const auto &localFiniteElement = localView_.tree().child(0).finiteElement();

      // Evaluate the shape functions
      std::vector<FieldVector<double, 1> > values;
      localFiniteElement.localBasis().evaluateFunction(local, values);

      FieldVector<RT,components_> result(0);

      for(size_t i=0; i<localFiniteElement.size(); i++)
        result.axpy(values[i][0], localHermiteCoefficients_[i]);

      return (RealTuple<RT, components_>)result;
    }

    /** \brief Evaluate the derivative of the function */
    DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, gridDim>& local) const
    {
      const auto &localFiniteElement = localView_.tree().child(0).finiteElement();

      const auto jacobianInverse = localView_.element().geometry().jacobianInverse(local);
      std::vector<FieldMatrix<double,1,gridDim> > jacobianValues;
      localFiniteElement.localBasis().evaluateJacobian(local, jacobianValues);
      for (size_t i = 0; i<jacobianValues.size(); i++)
        jacobianValues[i] = jacobianValues[i] * jacobianInverse;

      DerivativeType result(0);

      for(size_t i=0; i<localFiniteElement.size(); i++)
        for (unsigned int k = 0; k<components_; k++)
          for (unsigned int j = 0; j<gridDim; ++j)
            result[k][j] += (jacobianValues[i][0][j] * localHermiteCoefficients_[i][k]);

      return result;
    }

    /** \brief Evaluate the derivative of the function, if you happen to know the function value (much faster!)
     *        \param local Local coordinates in the reference element where to evaluate the derivative
     *        \param q Value of the local gfe function at 'local'.  If you provide something wrong here the result will be wrong, too!
     */
    DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, gridDim>& local,
                                      const Coefficient& q) const
    {
      return evaluateDerivative(local);
    }



    /** \brief Evaluate the discrete gradient operator (This quantity lives in the P2-rotation space) */
    DerivativeType evaluateDiscreteGradient(const Dune::FieldVector<ctype, gridDim>& local) const
    {
      //Get values of the P2-Basis functions on local point
      std::vector<FieldVector<double,1> > basisValues;
      rotationLFE_.localBasis().evaluateFunction(local, basisValues);

      DerivativeType discreteGradient(0);

      for (int k=0; k<components_; k++)
        for (int l=0; l<gridDim; l++)
          for (std::size_t i=0; i<rotationLFE_.size(); i++)
            discreteGradient[k][l]  += discreteJacobianCoefficients_[i][k][l]*basisValues[i];

      return discreteGradient;
    }


    /** \brief Evaluate the discrete hessian operator (This quantity lives in the P2-rotation space) */
    discreteHessianType evaluateDiscreteHessian(const Dune::FieldVector<ctype, gridDim>& local) const
    {
      // Get gradient values of the P2-Basis functions on local point
      const auto geometryJacobianInverse = localView_.element().geometry().jacobianInverse(local);
      std::vector<FieldMatrix<double,1,gridDim> > gradients;
      rotationLFE_.localBasis().evaluateJacobian(local, gradients);

      for (size_t i=0; i< gradients.size(); i++)
        gradients[i] = gradients[i] * geometryJacobianInverse;

      discreteHessianType discreteHessian(0);

      for (int k=0; k<components_; k++)
        for (int l=0; l<gridDim; l++)
          for (std::size_t i=0; i<rotationLFE_.size(); i++)
          {
            discreteHessian[k][l][0]  += discreteJacobianCoefficients_[i][k][l]*gradients[i][0][0];
            discreteHessian[k][l][1]  += discreteJacobianCoefficients_[i][k][l]*gradients[i][0][1];
          }

      return discreteHessian;
    }


    //! Obtain the grid view that the basis is defined on
    const GridView &gridView() const
    {
      return basis_.gridView();
    }

    void updateglobalIsometryCoefficients(const Coefficients& newCoefficients)
    {
      globalIsometryCoefficients_ = newCoefficients;
    }

    auto getDiscreteJacobianCoefficients()
    {
      return discreteJacobianCoefficients_;
    }

    /**
        Update the local hermite basis coefficient vector
     */
    void updateLocalHermiteCoefficients(const Coefficients& localIsometryCoefficients,const typename GridView::template Codim<0>::Entity& element)
    {
      localViewCoefficient_.bind(element);

      //Create a LocalProjected-Finite element from the local coefficients used for interpolation.
      auto P1LagrangeLFE = localViewCoefficient_.tree().finiteElement();
      LocalInterpolationRule localPBfunction;
      localPBfunction.bind(P1LagrangeLFE,localIsometryCoefficients);

      /**
          Interpolate into the local hermite space for each component.
       */
      const auto &localHermiteFiniteElement = localView_.tree().child(0).finiteElement();

      for(size_t k=0; k<components_; k++)
      {
        std::vector<RT> hermiteComponentCoefficients;
        Dune::GFE::Impl::LocalIsometryComponentFunction<double,LocalInterpolationRule> localIsometryComponentFunction(localPBfunction,k);

        localHermiteFiniteElement.localInterpolation().interpolate(localIsometryComponentFunction,hermiteComponentCoefficients);

        for(size_t i=0; i<localHermiteFiniteElement.size(); i++)
          localHermiteCoefficients_[i][k] = hermiteComponentCoefficients[i];
      }
    }

    /** Extract the local isometry coefficients. */
    void getLocalIsometryCoefficients(std::vector<Coefficient>& in,const typename GridView::template Codim<0>::Entity& element)
    {
      localViewCoefficient_.bind(element);

      in.resize(3);
      for (unsigned int vertex = 0; vertex<3; ++vertex)
      {
        size_t localIdx = localViewCoefficient_.tree().localIndex(vertex);
        size_t globalIdx = localViewCoefficient_.index(localIdx);
        in[vertex] = globalIsometryCoefficients_[globalIdx];
      }
    }

    /** \brief Return the representation LocalFiniteElement for the rotation space */
    auto const &getRotationFiniteElement() const
    {
      return rotationLFE_;
    }

    /** \brief Return the discrete Jacobian coefficients */
    BlockVector<FieldMatrix<RT, components_, gridDim> > getDiscreteJacobianCoefficients() const
    {
      return discreteJacobianCoefficients_;
    }

    /** \brief Get the i'th base coefficient. */
    Coefficients coefficient(int i) const
    {
      return globalIsometryCoefficients_[i];
    }

  private:

    /** \brief The Lagrange basis used to assign manifold-valued degrees of freedom */
    const CoefficientBasis& coefficientBasis_;

    /** \brief The hermite basis used to represent deformations */
    const DiscreteKirchhoffBasis& basis_;

    /** \brief LocalFiniteElement (P2-Lagrange) used to represent the discrete Jacobian (rotation) on the current element. */
    Dune::LagrangeSimplexLocalFiniteElement<ctype, double, gridDim, 2> rotationLFE_;

    /** \brief The coefficients of the discrete Gradient/Hessian operator */
    BlockVector<FieldMatrix<RT, components_, gridDim> > discreteJacobianCoefficients_;

    mutable typename DiscreteKirchhoffBasis::LocalView localView_;
    mutable typename CoefficientBasis::LocalView localViewCoefficient_;

    /** \brief The global coefficient vector */
    Coefficients& globalIsometryCoefficients_;

    static constexpr int localHermiteBasisSize_ = Dune::Functions::Impl::CubicHermiteLocalCoefficients<gridDim,true>::size();

    /** \brief The local coefficient vector of the hermite basis. */
    std::array<Dune::FieldVector<RT,components_>,localHermiteBasisSize_> localHermiteCoefficients_;

  };

}  // end namespace Dune::GFE
#endif   // DUNE_GFE_FUNCTIONS_DISCRETEKIRCHHOFFBENDINGISOMETRY_HH
