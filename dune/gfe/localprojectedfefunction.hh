#ifndef DUNE_GFE_LOCALPROJECTEDFEFUNCTION_HH
#define DUNE_GFE_LOCALPROJECTEDFEFUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/geometry/type.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/rigidbodymotion.hh>
#include <dune/gfe/linearalgebra.hh>

namespace Dune {

  namespace GFE {

    /** \brief Interpolate in an embedding Euclidean space, and project back onto the Riemannian manifold
     *
     * \tparam dim Dimension of the reference element
     * \tparam ctype Type used for coordinates on the reference element
     * \tparam LocalFiniteElement A Lagrangian finite element whose shape functions define the interpolation weights
     * \tparam TargetSpace The manifold that the function takes its values in
     */
    template <int dim, class ctype, class LocalFiniteElement, class TS>
    class LocalProjectedFEFunction
    {
    public:
      using TargetSpace=TS;
    private:
      typedef typename TargetSpace::ctype RT;

      typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
      static const int embeddedDim = EmbeddedTangentVector::dimension;

      static const int spaceDim = TargetSpace::TangentVector::dimension;

    public:

      /** \brief The type used for derivatives */
      typedef Dune::FieldMatrix<RT, embeddedDim, dim> DerivativeType;

      /** \brief Constructor
       * \param localFiniteElement A Lagrangian finite element that provides the interpolation points
       * \param coefficients Values of the function at the Lagrange points
       */
      LocalProjectedFEFunction(const LocalFiniteElement& localFiniteElement,
                               const std::vector<TargetSpace>& coefficients)
      : localFiniteElement_(localFiniteElement),
      coefficients_(coefficients)
      {
        assert(localFiniteElement_.localBasis().size() == coefficients_.size());
      }

      /** \brief Rebind the FEFunction to another TargetSpace */
      template<class U>
      struct rebind
      {
        using other = LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,U>;
      };

      /** \brief The number of Lagrange points */
      unsigned int size() const
      {
        return localFiniteElement_.localBasis().size();
      }

      /** \brief The type of the reference element */
      Dune::GeometryType type() const
      {
        return localFiniteElement_.type();
      }

      /** \brief Evaluate the function */
      TargetSpace evaluate(const Dune::FieldVector<ctype, dim>& local) const;

      /** \brief Evaluate the derivative of the function */
      DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const;

      /** \brief Evaluate the derivative of the function, if you happen to know the function value (much faster!)
       *        \param local Local coordinates in the reference element where to evaluate the derivative
       *        \param q Value of the local gfe function at 'local'.  If you provide something wrong here the result will be wrong, too!
       */
      DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local,
                                        const TargetSpace& q) const;

      /** \brief Get the i'th base coefficient. */
      TargetSpace coefficient(int i) const
      {
        return coefficients_[i];
      }
    private:

      /** \brief The scalar local finite element, which provides the weighting factors
       *        \todo We really only need the local basis
       */
      const LocalFiniteElement& localFiniteElement_;

      /** \brief The coefficient vector */
      std::vector<TargetSpace> coefficients_;

    };

    template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
    TargetSpace LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
    evaluate(const Dune::FieldVector<ctype, dim>& local) const
    {
      // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
      std::vector<Dune::FieldVector<ctype,1> > w;
      localFiniteElement_.localBasis().evaluateFunction(local,w);

      typename TargetSpace::CoordinateType c(0);
      for (size_t i=0; i<coefficients_.size(); i++)
        c.axpy(w[i][0], coefficients_[i].globalCoordinates());

      return TargetSpace::projectOnto(c);
    }

    template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
    typename LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::DerivativeType
    LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
    evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const
    {
      // the function value at the point where we are evaluating the derivative
      TargetSpace q = evaluate(local);

      // Actually compute the derivative
      return evaluateDerivative(local,q);
    }

    template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
    typename LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::DerivativeType
    LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
    evaluateDerivative(const Dune::FieldVector<ctype, dim>& local, const TargetSpace& q) const
    {
      // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
      std::vector<Dune::FieldVector<ctype,1> > w;
      localFiniteElement_.localBasis().evaluateFunction(local,w);

      std::vector<Dune::FieldMatrix<ctype,1,dim> > wDer;
      localFiniteElement_.localBasis().evaluateJacobian(local,wDer);

      typename TargetSpace::CoordinateType embeddedInterpolation(0);
      for (size_t i=0; i<coefficients_.size(); i++)
        embeddedInterpolation.axpy(w[i][0], coefficients_[i].globalCoordinates());

      Dune::FieldMatrix<RT,embeddedDim,dim> derivative(0);
      for (size_t i=0; i<embeddedDim; i++)
        for (size_t j=0; j<dim; j++)
          for (size_t k=0; k<coefficients_.size(); k++)
            derivative[i][j] += wDer[k][0][j] * coefficients_[k].globalCoordinates()[i];

      auto derivativeOfProjection = TargetSpace::derivativeOfProjection(embeddedInterpolation);

      return derivativeOfProjection*derivative;
    }

    /** \brief Interpolate in an embedding Euclidean space, and project back onto the Riemannian manifold -- specialization for SO(3)
     *
     * \tparam dim Dimension of the reference element
     * \tparam ctype Type used for coordinates on the reference element
     * \tparam LocalFiniteElement A Lagrangian finite element whose shape functions define the interpolation weights
     */
    template <int dim, class ctype, class LocalFiniteElement, class field_type>
    class LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,Rotation<field_type,3> >
    {
    public:
      typedef Rotation<field_type,3> TargetSpace;
    private:
      typedef typename TargetSpace::ctype RT;

      typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
      static const int embeddedDim = EmbeddedTangentVector::dimension;

      static const int spaceDim = TargetSpace::TangentVector::dimension;

      static FieldMatrix<field_type,3,3> polarFactor(const FieldMatrix<field_type,3,3>& matrix)
      {
        // Use Higham's method
        auto polar = matrix;
        for (size_t i=0; i<3; i++)
        {
          auto polarInvert = polar;
          polarInvert.invert();
          for (size_t j=0; j<polar.N(); j++)
            for (size_t k=0; k<polar.M(); k++)
              polar[j][k] = 0.5 * (polar[j][k] + polarInvert[k][j]);
        }

        return polar;
      }

      /**
       * \param A The argument of the projection
       * \param polar The image of the projection, i.e., the polar factor of A
       */
      static std::array<std::array<FieldMatrix<field_type,3,3>, 3>, 3> derivativeOfProjection(const FieldMatrix<field_type,3,3>& A,
                                                                       FieldMatrix<field_type,3,3>& polar)
      {
        std::array<std::array<FieldMatrix<field_type,3,3>, 3>, 3> result;

        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            for (int k=0; k<3; k++)
              for (int l=0; l<3; l++)
                result[i][j][k][l] = (i==k) and (j==l);

        polar = A;

        // Use Heron's method
        const size_t maxIterations = 3;
        for (size_t iteration=0; iteration<maxIterations; iteration++)
        {
          auto polarInvert = polar;
          polarInvert.invert();
          for (size_t i=0; i<polar.N(); i++)
            for (size_t j=0; j<polar.M(); j++)
              polar[i][j] = 0.5 * (polar[i][j] + polarInvert[j][i]);

          // Alternative name to align the code better with a description in a math text
          const auto& dQT = result;

          // Multiply from the right with Q^{-T}
          decltype(result) tmp2;
          for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
              for (int k=0; k<3; k++)
                for (int l=0; l<3; l++)
                  tmp2[i][j][k][l] = 0.0;

          for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
              for (int k=0; k<3; k++)
                for (int l=0; l<3; l++)
                  for (int m=0; m<3; m++)
                    for (int o=0; o<3; o++)
                      tmp2[i][j][k][l] += polarInvert[m][i] * dQT[o][m][k][l] * polarInvert[j][o];

          for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
              for (int k=0; k<3; k++)
                for (int l=0; l<3; l++)
                  result[i][j][k][l] = 0.5 * (result[i][j][k][l] - tmp2[i][j][k][l]);
        }

        return result;
      }


    public:

      /** \brief The type used for derivatives */
      typedef Dune::FieldMatrix<RT, embeddedDim, dim> DerivativeType;

      /** \brief Constructor
       * \param localFiniteElement A Lagrangian finite element that provides the interpolation points
       * \param coefficients Values of the function at the Lagrange points
       */
      LocalProjectedFEFunction(const LocalFiniteElement& localFiniteElement,
                               const std::vector<TargetSpace>& coefficients)
      : localFiniteElement_(localFiniteElement),
      coefficients_(coefficients)
      {
        assert(localFiniteElement_.localBasis().size() == coefficients_.size());
      }

      /** \brief The number of Lagrange points */
      unsigned int size() const
      {
        return localFiniteElement_.size();
      }

      /** \brief The type of the reference element */
      Dune::GeometryType type() const
      {
        return localFiniteElement_.type();
      }

      /** \brief Evaluate the function */
      TargetSpace evaluate(const Dune::FieldVector<ctype, dim>& local) const
      {
        Rotation<field_type,3> result;

        // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
        std::vector<Dune::FieldVector<ctype,1> > w;
        localFiniteElement_.localBasis().evaluateFunction(local,w);

        // Interpolate in R^{3x3}
        FieldMatrix<field_type,3,3> interpolatedMatrix(0);
        for (size_t i=0; i<coefficients_.size(); i++)
        {
          FieldMatrix<field_type,3,3> coefficientAsMatrix;
          coefficients_[i].matrix(coefficientAsMatrix);
          interpolatedMatrix.axpy(w[i][0], coefficientAsMatrix);
        }

        // Project back onto SO(3)
        result.set(polarFactor(interpolatedMatrix));

        return result;
      }

      /** \brief Evaluate the derivative of the function */
      DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const
      {
        // the function value at the point where we are evaluating the derivative
        TargetSpace q = evaluate(local);

        // Actually compute the derivative
        return evaluateDerivative(local,q);
      }

      /** \brief Evaluate the derivative of the function, if you happen to know the function value (much faster!)
       *        \param local Local coordinates in the reference element where to evaluate the derivative
       *        \param q Value of the local function at 'local'.  If you provide something wrong here the result will be wrong, too!
       */
      DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local,
                                        const TargetSpace& q) const
      {
        // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
        std::vector<Dune::FieldVector<ctype,1> > w;
        localFiniteElement_.localBasis().evaluateFunction(local,w);

        std::vector<Dune::FieldMatrix<ctype,1,dim> > wDer;
        localFiniteElement_.localBasis().evaluateJacobian(local,wDer);

        // Compute matrix representations for all coefficients (we only have them in quaternion representation)
        std::vector<Dune::FieldMatrix<field_type,3,3> > coefficientsAsMatrix(coefficients_.size());
        for (size_t i=0; i<coefficients_.size(); i++)
          coefficients_[i].matrix(coefficientsAsMatrix[i]);

        // Interpolate in R^{3x3}
        FieldMatrix<field_type,3,3> interpolatedMatrix(0);
        for (size_t i=0; i<coefficients_.size(); i++)
          interpolatedMatrix.axpy(w[i][0], coefficientsAsMatrix[i]);

        Tensor3<RT,dim,3,3> derivative(0);

        for (size_t dir=0; dir<dim; dir++)
          for (size_t i=0; i<3; i++)
            for (size_t j=0; j<3; j++)
              for (size_t k=0; k<coefficients_.size(); k++)
                derivative[dir][i][j] += wDer[k][0][dir] * coefficientsAsMatrix[k][i][j];

        FieldMatrix<field_type,3,3> polarFactor;
        auto derivativeOfProjection = this->derivativeOfProjection(interpolatedMatrix,polarFactor);

        Tensor3<field_type,dim,3,3> intermediateResult(0);

        for (size_t dir=0; dir<dim; dir++)
          for (size_t i=0; i<3; i++)
            for (size_t j=0; j<3; j++)
              for (size_t k=0; k<3; k++)
                for (size_t l=0; l<3; l++)
                  intermediateResult[dir][i][j] += derivativeOfProjection[i][j][k][l]*derivative[dir][k][l];

        // One more application of the chain rule: we need to go from orthogonal matrices to quaternions
        Tensor3<field_type,4,3,3> derivativeOfMatrixToQuaternion = Rotation<field_type,3>::derivativeOfMatrixToQuaternion(polarFactor);

        DerivativeType result(0);

        for (size_t dir0=0; dir0<4; dir0++)
          for (size_t dir1=0; dir1<dim; dir1++)
            for (size_t i=0; i<3; i++)
              for (size_t j=0; j<3; j++)
                result[dir0][dir1] += derivativeOfMatrixToQuaternion[dir0][i][j] * intermediateResult[dir1][i][j];

        return result;
      }

      /** \brief Get the i'th base coefficient. */
      TargetSpace coefficient(int i) const
      {
        return coefficients_[i];
      }
    private:

      /** \brief The scalar local finite element, which provides the weighting factors
       *        \todo We really only need the local basis
       */
      const LocalFiniteElement& localFiniteElement_;

      /** \brief The coefficient vector */
      std::vector<TargetSpace> coefficients_;

    };


    /** \brief Interpolate in an embedding Euclidean space, and project back onto the Riemannian manifold -- specialization for R^3 x SO(3)
     *
     * \tparam dim Dimension of the reference element
     * \tparam ctype Type used for coordinates on the reference element
     * \tparam LocalFiniteElement A Lagrangian finite element whose shape functions define the interpolation weights
     */
    template <int dim, class ctype, class LocalFiniteElement, class field_type>
    class LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,RigidBodyMotion<field_type,3> >
    {
    public:
      typedef RigidBodyMotion<field_type,3> TargetSpace;
    private:
      typedef typename TargetSpace::ctype RT;

      typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
      static const int embeddedDim = EmbeddedTangentVector::dimension;

      static const int spaceDim = TargetSpace::TangentVector::dimension;

    public:

      /** \brief The type used for derivatives */
      typedef Dune::FieldMatrix<RT, embeddedDim, dim> DerivativeType;

      /** \brief Constructor
       * \param localFiniteElement A Lagrangian finite element that provides the interpolation points
       * \param coefficients Values of the function at the Lagrange points
       */
      LocalProjectedFEFunction(const LocalFiniteElement& localFiniteElement,
                               const std::vector<TargetSpace>& coefficients)
      : localFiniteElement_(localFiniteElement),
        translationCoefficients_(coefficients.size())
      {
        assert(localFiniteElement.localBasis().size() == coefficients.size());

        for (size_t i=0; i<coefficients.size(); i++)
            translationCoefficients_[i] = coefficients[i].r;

        std::vector<Rotation<field_type,3> > orientationCoefficients(coefficients.size());
        for (size_t i=0; i<coefficients.size(); i++)
            orientationCoefficients[i] = coefficients[i].q;

        orientationFunction_ = std::make_unique<LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,Rotation<field_type,3> > > (localFiniteElement,orientationCoefficients);
      }

      /** \brief Rebind the FEFunction to another TargetSpace */
      template<class U>
      struct rebind
      {
        using other = LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,U>;
      };

      /** \brief The number of Lagrange points */
      unsigned int size() const
      {
        return localFiniteElement_.size();
      }

      /** \brief The type of the reference element */
      Dune::GeometryType type() const
      {
        return localFiniteElement_.type();
      }

      /** \brief Evaluate the function */
      TargetSpace evaluate(const Dune::FieldVector<ctype, dim>& local) const
      {
        RigidBodyMotion<field_type,3> result;

        // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
        std::vector<Dune::FieldVector<ctype,1> > w;
        localFiniteElement_.localBasis().evaluateFunction(local,w);

        result.r = 0;
        for (size_t i=0; i<w.size(); i++)
            result.r.axpy(w[i][0], translationCoefficients_[i]);

        result.q = orientationFunction_->evaluate(local);

        return result;
      }

      /** \brief Evaluate the derivative of the function */
      DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const
      {
        // the function value at the point where we are evaluating the derivative
        TargetSpace q = evaluate(local);

        // Actually compute the derivative
        return evaluateDerivative(local,q);
      }

      /** \brief Evaluate the derivative of the function, if you happen to know the function value (much faster!)
       *        \param local Local coordinates in the reference element where to evaluate the derivative
       *        \param q Value of the local function at 'local'.  If you provide something wrong here the result will be wrong, too!
       */
      DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local,
                                        const TargetSpace& q) const
      {
        DerivativeType result(0);

        // get translation part
        std::vector<Dune::FieldMatrix<ctype,1,dim> > sfDer(translationCoefficients_.size());
        localFiniteElement_.localBasis().evaluateJacobian(local, sfDer);

        for (size_t i=0; i<translationCoefficients_.size(); i++)
            for (int j=0; j<3; j++)
                result[j].axpy(translationCoefficients_[i][j], sfDer[i][0]);

        // get orientation part
        Dune::FieldMatrix<field_type,4,dim> qResult = orientationFunction_->evaluateDerivative(local,q.q);

        for (int i=0; i<4; i++)
            for (int j=0; j<dim; j++)
                result[3+i][j] = qResult[i][j];

        return result;
      }

      /** \brief Get the i'th base coefficient. */
      TargetSpace coefficient(int i) const
      {
        return TargetSpace(translationCoefficients_[i],orientationFunction_->coefficient(i));
      }
    private:

      /** \brief The scalar local finite element, which provides the weighting factors
       *        \todo We really only need the local basis
       */
      const LocalFiniteElement& localFiniteElement_;

      std::vector<Dune::FieldVector<field_type,3> > translationCoefficients_;

      std::unique_ptr<LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,Rotation<field_type, 3> > > orientationFunction_;


    };

  }

}
#endif
