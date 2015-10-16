#ifndef DUNE_GFE_LOCALPROJECTEDFEFUNCTION_HH
#define DUNE_GFE_LOCALPROJECTEDFEFUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/geometry/type.hh>

#include <dune/gfe/rotation.hh>
#include <dune/gfe/svd.hh>
#include <dune/gfe/linearalgebra.hh>

namespace Dune {

template< class K, int m, int n, int p >
Dune::FieldMatrix< K, m, p > operator* ( const Dune::FieldMatrix< K, m, n > &A, const Dune::FieldMatrix< K, n, p > &B)
{
    typedef typename Dune::FieldMatrix< K, m, p > :: size_type size_type;
    Dune::FieldMatrix< K, m, p > ret;

    for( size_type i = 0; i < m; ++i ) {

        for( size_type j = 0; j < p; ++j ) {
            ret[ i ][ j ] = K( 0 );
            for( size_type k = 0; k < n; ++k )
                ret[ i ][ j ] += A[ i ][ k ] * B[ k ][ j ];
        }
    }
    return ret;
}
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

      return TargetSpace(c);
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

      typename TargetSpace::DerivativeOfProjection derivativeOfProjection = TargetSpace::derivativeOfProjection(embeddedInterpolation);

      typename LocalProjectedFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::DerivativeType result;

      for (size_t i=0; i<result.N(); i++)
        for (size_t j=0; j<result.M(); j++)
        {
          result[i][j] = 0;
          for (size_t k=0; k<derivativeOfProjection.M(); k++)
            result[i][j] += derivativeOfProjection[i][k]*derivative[k][j];
        }

      return result;
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

      static FieldMatrix<field_type,3,3> transpose(const FieldMatrix<field_type,3,3>& matrix)
      {
        FieldMatrix<field_type,3,3> result;

        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            result[i][j] = matrix[j][i];

        return result;
      }

      static FieldMatrix<field_type,3,3> polarFactor(const FieldMatrix<field_type,3,3>& matrix)
      {
        FieldVector<field_type,3> W;
        FieldMatrix<field_type,3,3> V, VT;
        auto polar = matrix;

        // returns a decomposition U W VT, where U is returned in the first argument
        svdcmp<field_type,3,3>(polar, W, V);

        // \todo Merge this transposition with the following multiplication
        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            VT[i][j] = V[j][i];

        polar.rightmultiply(VT);
        return polar;
      }

      /**
       * \param A The argument of the projection
       * \param value The image of the projection
       */
      static std::array<std::array<FieldMatrix<field_type,3,3>, 3>, 3> derivativeOfProjection(const FieldMatrix<field_type,3,3> A,
                                                                       Rotation<field_type,3> value)
      {
        std::array<std::array<FieldMatrix<field_type,3,3>, 3>, 3> result;

        /////////////////////////////////////////////////////////////////////////////////
        //  Step I: Compute the singular value decomposition of A
        /////////////////////////////////////////////////////////////////////////////////

        FieldVector<field_type,3> sigma;
        FieldMatrix<field_type,3,3> V, VT;
        FieldMatrix<field_type,3,3> U = A;

        // returns a decomposition U diag(sigma) VT, where U is returned in the first argument
        svdcmp<field_type,3,3>(U, sigma, V);

        // Compute VT from V (svdcmp returns the latter)
        for (int i=0; i<3; i++)
          for (int j=0; j<3; j++)
            VT[i][j] = V[j][i];

        /////////////////////////////////////////////////////////////////////////////////
        //  Step II: Compute the derivative direction X
        /////////////////////////////////////////////////////////////////////////////////

        for (int dir0=0; dir0<3; dir0++)
        {
          for (int dir1=0; dir1<3; dir1++)
          {
            FieldMatrix<field_type,3,3> X(0);
            X[dir0][dir1] = 1.0;

            /////////////////////////////////////////////////////////////////////////////////
            //  Step ???: Markus' magic formula
            /////////////////////////////////////////////////////////////////////////////////

            FieldMatrix<field_type,3,3> UTXV = transpose(U)*X*V;

            FieldMatrix<field_type,3,3> center;
            center[0][0] = 0.0;
            center[0][1] = (UTXV[0][1] - UTXV[1][0]) / (sigma[0] + sigma[1]);
            center[0][2] = (UTXV[0][2] - UTXV[2][0]) / (sigma[0] + sigma[2]);

            center[1][0] = (UTXV[1][0] - UTXV[0][1]) / (sigma[1] + sigma[0]);
            center[1][1] = 0.0;
            center[1][2] = (UTXV[1][2] - UTXV[2][1]) / (sigma[1] + sigma[2]);

            center[2][0] = (UTXV[2][0] - UTXV[0][2]) / (sigma[2] + sigma[0]);
            center[2][1] = (UTXV[2][1] - UTXV[1][2]) / (sigma[2] + sigma[1]);
            center[2][2] = 0.0;

            // Compute U center VT
            FieldMatrix<field_type,3,3> PAX = U*center*VT;

            // Copy into result array
            for (int i=0; i<3; i++)
              for (int j=0; j<3; j++)
                result[i][j][dir0][dir1] = PAX[i][j];

          }
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

        auto polarFactor = this->polarFactor(interpolatedMatrix);

        Tensor3<RT,dim,3,3> derivative(0);

        for (size_t dir=0; dir<dim; dir++)
          for (size_t i=0; i<3; i++)
            for (size_t j=0; j<3; j++)
              for (size_t k=0; k<coefficients_.size(); k++)
                derivative[dir][i][j] += wDer[k][0][dir] * coefficientsAsMatrix[k][i][j];

        auto derivativeOfProjection = this->derivativeOfProjection(interpolatedMatrix,q);

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

  }

}
#endif
