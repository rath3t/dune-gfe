#ifndef LOCAL_GEODESIC_FE_FUNCTION_HH
#define LOCAL_GEODESIC_FE_FUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/geometry/type.hh>

#include <dune/gfe/averagedistanceassembler.hh>
#include <dune/gfe/targetspacertrsolver.hh>
#include <dune/gfe/rigidbodymotion.hh>

#include <dune/gfe/tensor3.hh>
#include <dune/gfe/tensorssd.hh>
#include <dune/gfe/linearalgebra.hh>

// forward declaration
template <class LocalFiniteElement, class TargetSpace>
class LocalGfeTestFunctionBasis;

/** \brief A function defined by simplicial geodesic interpolation
           from the reference element to a Riemannian manifold.

\tparam dim Dimension of the reference element
\tparam ctype Type used for coordinates on the reference element
\tparam LocalFiniteElement A Lagrangian finite element whose shape functions define the interpolation weights
\tparam TargetSpace The manifold that the function takes its values in
*/
template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
class LocalGeodesicFEFunction
{
    typedef typename TargetSpace::ctype RT;

    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
    static const int embeddedDim = EmbeddedTangentVector::dimension;

    static const int spaceDim = TargetSpace::TangentVector::dimension;

    friend class LocalGfeTestFunctionBasis<LocalFiniteElement,TargetSpace>;
public:

    /** \brief The type used for derivatives */
    typedef Dune::FieldMatrix<RT, embeddedDim, dim> DerivativeType;

    /** \brief The type used for derivatives of the gradient with respect to coefficients */
    typedef Tensor3<RT,embeddedDim,embeddedDim,dim> DerivativeOfGradientWRTCoefficientType;

    /** \brief Constructor
     * \param localFiniteElement A Lagrangian finite element that provides the interpolation points
     * \param coefficients Values of the function at the Lagrange points
     */
    LocalGeodesicFEFunction(const LocalFiniteElement& localFiniteElement,
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
        \param local Local coordinates in the reference element where to evaluate the derivative
        \param q Value of the local gfe function at 'local'.  If you provide something wrong here the result will be wrong, too!
     */
    DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local,
                                      const TargetSpace& q) const;

    /** \brief For debugging: Evaluate the derivative of the function using a finite-difference approximation*/
    DerivativeType evaluateDerivativeFD(const Dune::FieldVector<ctype, dim>& local) const;

    /** \brief Evaluate the derivative of the function value with respect to a coefficient */
    void evaluateDerivativeOfValueWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                                 int coefficient,
                                                 Dune::FieldMatrix<RT,embeddedDim,embeddedDim>& derivative) const;

    /** \brief Evaluate the derivative of the function value with respect to a coefficient */
    void evaluateFDDerivativeOfValueWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                                 int coefficient,
                                                 Dune::FieldMatrix<RT,embeddedDim,embeddedDim>& derivative) const;

    /** \brief Evaluate the derivative of the gradient of the function with respect to a coefficient */
    void evaluateDerivativeOfGradientWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                                    int coefficient,
                                                    DerivativeOfGradientWRTCoefficientType& result) const;

    /** \brief Evaluate the derivative of the gradient of the function with respect to a coefficient */
    void evaluateFDDerivativeOfGradientWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                                    int coefficient,
                                                    DerivativeOfGradientWRTCoefficientType& result) const;

    /** \brief Get the i'th base coefficient. */
    TargetSpace coefficient(int i) const
    {
        return coefficients_[i];
    }
private:

    static Dune::FieldMatrix<RT,embeddedDim,embeddedDim> pseudoInverse(const Dune::FieldMatrix<RT,embeddedDim,embeddedDim>& dFdq,
                                                                           const TargetSpace& q)
    {
        const int shortDim = TargetSpace::TangentVector::dimension;

        // the orthonormal frame
        Dune::FieldMatrix<RT,shortDim,embeddedDim> O = q.orthonormalFrame();

        // compute A = O dFDq O^T
        Dune::FieldMatrix<RT,shortDim,shortDim> A;
        for (int i=0; i<shortDim; i++)
            for (int j=0; j<shortDim; j++) {
                A[i][j] = 0;
                for (int k=0; k<embeddedDim; k++)
                    for (int l=0; l<embeddedDim; l++)
                        A[i][j] += O[i][k]*dFdq[k][l]*O[j][l];
            }

        A.invert();

        Dune::FieldMatrix<RT,embeddedDim,embeddedDim> result;
        for (int i=0; i<embeddedDim; i++)
            for (int j=0; j<embeddedDim; j++) {
                result[i][j] = 0;
                for (int k=0; k<shortDim; k++)
                    for (int l=0; l<shortDim; l++)
                        result[i][j] += O[k][i]*A[k][l]*O[l][j];
            }

        return result;
    }

    /** \brief Compute derivate of F(w,q) (the derivative of the weighted distance fctl) wrt to w */
    Dune::Matrix<Dune::FieldMatrix<RT,1,1> > computeDFdw(const TargetSpace& q) const
    {
        Dune::Matrix<Dune::FieldMatrix<RT,1,1> > dFdw(embeddedDim,localFiniteElement_.localBasis().size());
        for (size_t i=0; i<localFiniteElement_.localBasis().size(); i++) {
            Dune::FieldVector<RT,embeddedDim> tmp = TargetSpace::derivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i], q);
            for (int j=0; j<embeddedDim; j++)
                dFdw[j][i] = tmp[j];
        }
        return dFdw;
    }

    Tensor3<RT,embeddedDim,embeddedDim,embeddedDim> computeDqDqF(const std::vector<Dune::FieldVector<ctype,1> >& w, const TargetSpace& q) const
    {
        Tensor3<RT,embeddedDim,embeddedDim,embeddedDim> result;
        result = Tensor3<RT,embeddedDim,embeddedDim,embeddedDim>(RT(0));
        for (size_t i=0; i<w.size(); i++)
            result.axpy(w[i][0], TargetSpace::thirdDerivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i],q));
        return result;
    }

    /** \brief The scalar local finite element, which provides the weighting factors
        \todo We really only need the local basis
     */
    const LocalFiniteElement& localFiniteElement_;

    /** \brief The coefficient vector */
    std::vector<TargetSpace> coefficients_;

};

template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
TargetSpace LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
evaluate(const Dune::FieldVector<ctype, dim>& local) const
{
    // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
    std::vector<Dune::FieldVector<ctype,1> > w;
    localFiniteElement_.localBasis().evaluateFunction(local,w);

    // The energy functional whose mimimizer is the value of the geodesic interpolation
    AverageDistanceAssembler<TargetSpace> assembler(coefficients_, w);

    TargetSpaceRiemannianTRSolver<TargetSpace> solver;

    solver.setup(&assembler,
                 coefficients_[0],   // initial iterate
                 1e-14,    // tolerance
                 100,      // maxTrustRegionSteps
                 2,       // initial trust region radius
                 100,      // inner iterations
                 1e-14     // inner tolerance
                 );

    solver.solve();

    return solver.getSol();
}

template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
typename LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::DerivativeType
LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const
{
    // the function value at the point where we are evaluating the derivative
    TargetSpace q = evaluate(local);

    // Actually compute the derivative
    return evaluateDerivative(local,q);
}

template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
typename LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::DerivativeType
LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
evaluateDerivative(const Dune::FieldVector<ctype, dim>& local, const TargetSpace& q) const
{
    Dune::FieldMatrix<RT, embeddedDim, dim> result;

#if 0  // this is probably faster than the general implementation, but we leave it out for testing purposes
    if (dim==1) {

        EmbeddedTangentVector tmp = TargetSpace::interpolateDerivative(coefficients_[0], coefficients_[1], local[0]);

        for (int i=0; i<embeddedDim; i++)
            result[i][0] = tmp[i];

    }
#endif

    // ////////////////////////////////////////////////////////////////////////
    //  The derivative is evaluated using the implicit function theorem.
    //  Hence we need to solve a small system of linear equations.
    // ////////////////////////////////////////////////////////////////////////

    // the matrix that turns coordinates on the reference simplex into coordinates on the standard simplex
    std::vector<Dune::FieldMatrix<ctype,1,dim> > BNested(coefficients_.size());
    localFiniteElement_.localBasis().evaluateJacobian(local, BNested);
    Dune::Matrix<Dune::FieldMatrix<ctype,1,1> > B(coefficients_.size(), dim);
    for (size_t i=0; i<coefficients_.size(); i++)
        for (size_t j=0; j<dim; j++)
            B[i][j] = BNested[i][0][j];

    // compute negative derivative of F(w,q) (the derivative of the weighted distance fctl) wrt to w
    Dune::Matrix<Dune::FieldMatrix<RT,1,1> > dFdw = computeDFdw(q);
    dFdw *= -1;

    // multiply the two previous matrices: the result is the right hand side
    Dune::Matrix<Dune::FieldMatrix<RT,1,1> > RHS = dFdw * B;

    // the actual system matrix
    std::vector<Dune::FieldVector<ctype,1> > w;
    localFiniteElement_.localBasis().evaluateFunction(local, w);

    AverageDistanceAssembler<TargetSpace> assembler(coefficients_, w);

    Dune::FieldMatrix<RT,embeddedDim,embeddedDim> dFdq(0);
    assembler.assembleEmbeddedHessian(q,dFdq);

    // ////////////////////////////////////
    //   solve the system
    //
    // We want to solve
    //     dFdq * x = rhs
    // However as dFdq is defined in the embedding space it has a positive rank.
    // Hence we use the orthonormal frame at q to get rid of its kernel.
    // That means we instead solve
    // O dFdq O^T O x = O rhs
    // ////////////////////////////////////

    const int shortDim = TargetSpace::TangentVector::dimension;

    // the orthonormal frame
    Dune::FieldMatrix<RT,shortDim,embeddedDim> O = q.orthonormalFrame();

    // compute A = O dFDq O^T
    Dune::FieldMatrix<RT,shortDim,shortDim> A;
    for (int i=0; i<shortDim; i++)
        for (int j=0; j<shortDim; j++) {
            A[i][j] = 0;
            for (int k=0; k<embeddedDim; k++)
                for (int l=0; l<embeddedDim; l++)
                    A[i][j] += O[i][k]*dFdq[k][l]*O[j][l];
        }

    for (int i=0; i<dim; i++) {

        Dune::FieldVector<RT,embeddedDim> rhs;
        for (int j=0; j<embeddedDim; j++)
            rhs[j] = RHS[j][i];

        Dune::FieldVector<RT,shortDim> shortRhs;
        O.mv(rhs,shortRhs);

        Dune::FieldVector<RT,shortDim> shortX;
        A.solve(shortX,shortRhs);

        Dune::FieldVector<RT,embeddedDim> x;
        O.mtv(shortX,x);

        for (int j=0; j<embeddedDim; j++)
            result[j][i] = x[j];

    }

    return result;
}

template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
typename LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::DerivativeType
LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
evaluateDerivativeFD(const Dune::FieldVector<ctype, dim>& local) const
{
    double eps = 1e-6;

    Dune::FieldMatrix<ctype, embeddedDim, dim> result;

    for (int i=0; i<dim; i++) {

        Dune::FieldVector<ctype, dim> forward  = local;
        Dune::FieldVector<ctype, dim> backward = local;

        forward[i]  += eps;
        backward[i] -= eps;

        EmbeddedTangentVector fdDer = evaluate(forward).globalCoordinates() - evaluate(backward).globalCoordinates();
        fdDer /= 2*eps;

        for (int j=0; j<embeddedDim; j++)
            result[j][i] = fdDer[j];

    }

    return result;
}

template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
void LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
evaluateDerivativeOfValueWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                        int coefficient,
                                        Dune::FieldMatrix<RT,embeddedDim,embeddedDim>& result) const
{
    // the function value at the point where we are evaluating the derivative
    TargetSpace q = evaluate(local);

    // dFdq
    std::vector<Dune::FieldVector<ctype,1> > w;
    localFiniteElement_.localBasis().evaluateFunction(local,w);

    AverageDistanceAssembler<TargetSpace> assembler(coefficients_, w);

    Dune::FieldMatrix<RT,embeddedDim,embeddedDim> dFdq(0);
    assembler.assembleEmbeddedHessian(q,dFdq);

    const int shortDim = TargetSpace::TangentVector::dimension;

    // the orthonormal frame
    Dune::FieldMatrix<RT,shortDim,embeddedDim> O = q.orthonormalFrame();

    // compute A = O dFDq O^T
    Dune::FieldMatrix<RT,shortDim,shortDim> A;
    for (int i=0; i<shortDim; i++)
        for (int j=0; j<shortDim; j++) {
            A[i][j] = 0;
            for (int k=0; k<embeddedDim; k++)
                for (int l=0; l<embeddedDim; l++)
                    A[i][j] += O[i][k]*dFdq[k][l]*O[j][l];
        }

    //
    Dune::FieldMatrix<RT,embeddedDim,embeddedDim> rhs = TargetSpace::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(coefficients_[coefficient], q);
    rhs *= -w[coefficient];

    for (int i=0; i<embeddedDim; i++) {

        Dune::FieldVector<RT,shortDim> shortRhs;
        O.mv(rhs[i],shortRhs);

        Dune::FieldVector<RT,shortDim> shortX;
        A.solve(shortX,shortRhs);

        O.mtv(shortX,result[i]);

    }

}

template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
void LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
evaluateFDDerivativeOfValueWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                          int coefficient,
                                          Dune::FieldMatrix<RT,embeddedDim,embeddedDim>& result) const
{
    double eps = 1e-6;

    // the function value at the point where we are evaluating the derivative
    const Dune::FieldMatrix<RT,spaceDim,embeddedDim> B = coefficients_[coefficient].orthonormalFrame();

    Dune::FieldMatrix<RT,spaceDim,embeddedDim> interimResult;

    std::vector<TargetSpace> cornersPlus  = coefficients_;
    std::vector<TargetSpace> cornersMinus = coefficients_;

    for (int j=0; j<spaceDim; j++) {

        typename TargetSpace::EmbeddedTangentVector forwardVariation = B[j];
        forwardVariation *= eps;
        typename TargetSpace::EmbeddedTangentVector backwardVariation = B[j];
        backwardVariation *= -eps;

        cornersPlus [coefficient] = TargetSpace::exp(coefficients_[coefficient], forwardVariation);
        cornersMinus[coefficient] = TargetSpace::exp(coefficients_[coefficient], backwardVariation);

        LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace> fPlus(localFiniteElement_,cornersPlus);
        LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace> fMinus(localFiniteElement_,cornersMinus);

        TargetSpace hPlus  = fPlus.evaluate(local);
        TargetSpace hMinus = fMinus.evaluate(local);

        interimResult[j]  = hPlus.globalCoordinates();
        interimResult[j] -= hMinus.globalCoordinates();
        interimResult[j] /= 2*eps;

    }

    for (int i=0; i<embeddedDim; i++)
        for (int j=0; j<embeddedDim; j++) {
            result[i][j] = 0;
            for (int k=0; k<spaceDim; k++)
                result[i][j] += B[k][i]*interimResult[k][j];
        }

}


template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
void LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
evaluateDerivativeOfGradientWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                           int coefficient,
                                           DerivativeOfGradientWRTCoefficientType& result) const
{
    // the function value at the point where we are evaluating the derivative
    TargetSpace q = evaluate(local);

    // the matrix that turns coordinates on the reference simplex into coordinates on the standard simplex
    std::vector<Dune::FieldMatrix<ctype,1,dim> > BNested(coefficients_.size());
    localFiniteElement_.localBasis().evaluateJacobian(local, BNested);
    Dune::Matrix<Dune::FieldMatrix<RT,1,1> > B(coefficients_.size(), dim);
    for (size_t i=0; i<coefficients_.size(); i++)
        for (size_t j=0; j<dim; j++)
            B[i][j] = BNested[i][0][j];

    // the actual system matrix
    std::vector<Dune::FieldVector<ctype,1> > w;
    localFiniteElement_.localBasis().evaluateFunction(local,w);

    AverageDistanceAssembler<TargetSpace> assembler(coefficients_, w);

    Dune::FieldMatrix<RT,embeddedDim,embeddedDim> dFdq(0);
    assembler.assembleEmbeddedHessian(q,dFdq);


    Dune::FieldMatrix<RT,embeddedDim,embeddedDim> mixedDerivative = TargetSpace::secondDerivativeOfDistanceSquaredWRTFirstAndSecondArgument(coefficients_[coefficient], q);
    TensorSSD<RT,embeddedDim,embeddedDim> dvDwF(coefficients_.size());
    dvDwF = 0;
    for (int i=0; i<embeddedDim; i++)
        for (int j=0; j<embeddedDim; j++)
            dvDwF(i, j, coefficient) = mixedDerivative[i][j];


    // dFDq is not invertible, if the target space is embedded into a higher-dimensional
    // Euclidean space.  Therefore we use its pseudo inverse.  I don't think that is the
    // best way, though.
    Dune::FieldMatrix<RT,embeddedDim,embeddedDim> dFdqPseudoInv = pseudoInverse(dFdq,q);

    //
    Tensor3<RT,embeddedDim,embeddedDim,embeddedDim> dvDqF
       =  TargetSpace::thirdDerivativeOfDistanceSquaredWRTFirst1AndSecond2Argument(coefficients_[coefficient], q);

    dvDqF = RT(w[coefficient]) * dvDqF;

    // Put it all together
    // dvq[i][j] = \partial q_j / \partial v_i
    Dune::FieldMatrix<RT,embeddedDim,embeddedDim> dvq;
    evaluateDerivativeOfValueWRTCoefficient(local,coefficient,dvq);

    Dune::FieldMatrix<RT, embeddedDim, dim> derivative = evaluateDerivative(local);

    Tensor3<RT, embeddedDim,embeddedDim,embeddedDim> dqdqF;
    dqdqF = computeDqDqF(w,q);

    TensorSSD<RT, embeddedDim,embeddedDim> dqdwF(coefficients_.size());

    for (size_t k=0; k<coefficients_.size(); k++) {
        Dune::FieldMatrix<RT,embeddedDim,embeddedDim> hesse = TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(coefficients_[k], q);
        for (int i=0; i<embeddedDim; i++)
            for (int j=0; j<embeddedDim; j++)
                dqdwF(i, j, k) = hesse[i][j];
    }

    TensorSSD<RT, embeddedDim,embeddedDim> dqdwF_times_dvq(coefficients_.size());
    for (int i=0; i<embeddedDim; i++)
        for (int j=0; j<embeddedDim; j++)
            for (size_t k=0; k<coefficients_.size(); k++) {
                dqdwF_times_dvq(i, j, k) = 0;
                for (int l=0; l<embeddedDim; l++)
                    dqdwF_times_dvq(i, j, k) += dqdwF(l, j, k) * dvq[i][l];
            }

    Tensor3<RT,embeddedDim,embeddedDim,dim> foo;
    foo =  -1 * dvDqF*derivative - (dvq*dqdqF)*derivative;
    TensorSSD<RT,embeddedDim,embeddedDim> bar(dim);
    bar = dvDwF * B + dqdwF_times_dvq*B;

    for (int i=0; i<embeddedDim; i++)
        for (int j=0; j<embeddedDim; j++)
            for (int k=0; k<dim; k++)
                foo[i][j][k] -= bar(i, j, k);

    result = RT(0);
    for (int i=0; i<embeddedDim; i++)
        for (int j=0; j<embeddedDim; j++)
            for (int k=0; k<dim; k++)
                for (int l=0; l<embeddedDim; l++)
                    result[i][j][k] += dFdqPseudoInv[j][l] * foo[i][l][k];

}


template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
void LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
evaluateFDDerivativeOfGradientWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                           int coefficient,
                                           DerivativeOfGradientWRTCoefficientType& result) const
{
    double eps = 1e-6;

    // loop over the different partial derivatives
    for (int j=0; j<embeddedDim; j++) {

        std::vector<TargetSpace> cornersPlus  = coefficients_;
        std::vector<TargetSpace> cornersMinus = coefficients_;
        typename TargetSpace::CoordinateType aPlus  = coefficients_[coefficient].globalCoordinates();
        typename TargetSpace::CoordinateType aMinus = coefficients_[coefficient].globalCoordinates();
        aPlus[j]  += eps;
        aMinus[j] -= eps;
        cornersPlus[coefficient]  = TargetSpace(aPlus);
        cornersMinus[coefficient] = TargetSpace(aMinus);
        LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace> fPlus(localFiniteElement_,cornersPlus);
        LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace> fMinus(localFiniteElement_,cornersMinus);

        Dune::FieldMatrix<RT,embeddedDim,dim> hPlus  = fPlus.evaluateDerivative(local);
        Dune::FieldMatrix<RT,embeddedDim,dim> hMinus = fMinus.evaluateDerivative(local);

        result[j]  = hPlus;
        result[j] -= hMinus;
        result[j] /= 2*eps;

    }

    for (int j=0; j<embeddedDim; j++) {

        TargetSpace q = evaluate(local);
        Dune::FieldVector<RT,embeddedDim> foo;
        for (int l=0; l<dim; l++) {

            for (int k=0; k<embeddedDim; k++)
                foo[k] = result[j][k][l];

            foo = q.projectOntoTangentSpace(foo);

            for (int k=0; k<embeddedDim; k++)
                result[j][k][l] = foo[k];

        }

    }

}


/** \brief A function defined by simplicial geodesic interpolation
           from the reference element to a RigidBodyMotion.

   This is a specialization for speeding up the code.
   We use that a RigidBodyMotion is a product manifold.

\tparam dim Dimension of the reference element
\tparam ctype Type used for coordinates on the reference element
*/
template <int dim, class ctype, class LocalFiniteElement>
class LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,RigidBodyMotion<double,3> >
{
    typedef RigidBodyMotion<double,3> TargetSpace;

    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
    static const int embeddedDim = EmbeddedTangentVector::dimension;

    static const int spaceDim = TargetSpace::TangentVector::dimension;

public:

    /** \brief The type used for derivatives */
    typedef Dune::FieldMatrix<ctype, embeddedDim, dim> DerivativeType;

    /** \brief The type used for derivatives of the gradient with respect to coefficients */
    typedef Tensor3<ctype,embeddedDim,embeddedDim,dim> DerivativeOfGradientWRTCoefficientType;

    /** \brief Constructor */
    LocalGeodesicFEFunction(const LocalFiniteElement& localFiniteElement,
                            const std::vector<TargetSpace>& coefficients)
    : localFiniteElement_(localFiniteElement),
      translationCoefficients_(coefficients.size())
    {
        assert(localFiniteElement.localBasis().size() == coefficients.size());

        for (size_t i=0; i<coefficients.size(); i++)
            translationCoefficients_[i] = coefficients[i].r;

        std::vector<Rotation<ctype,3> > orientationCoefficients(coefficients.size());
        for (size_t i=0; i<coefficients.size(); i++)
            orientationCoefficients[i] = coefficients[i].q;

        orientationFEFunction_ = std::auto_ptr<LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,Rotation<double,3> > > (new LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,Rotation<double,3> >(localFiniteElement,orientationCoefficients));

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
    TargetSpace evaluate(const Dune::FieldVector<ctype, dim>& local) const
    {
        TargetSpace result;

        // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
        std::vector<Dune::FieldVector<ctype,1> > w;
        localFiniteElement_.localBasis().evaluateFunction(local,w);

        result.r = 0;
        for (size_t i=0; i<w.size(); i++)
            result.r.axpy(w[i][0], translationCoefficients_[i]);

        result.q = orientationFEFunction_->evaluate(local);
        return result;
    }

    /** \brief Evaluate the derivative of the function */
    DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const
    {
        Dune::FieldMatrix<ctype, embeddedDim, dim> result(0);

        // get translation part
        std::vector<Dune::FieldMatrix<ctype,1,dim> > sfDer(translationCoefficients_.size());
        localFiniteElement_.localBasis().evaluateJacobian(local, sfDer);

        for (size_t i=0; i<translationCoefficients_.size(); i++)
            for (int j=0; j<3; j++)
                result[j].axpy(translationCoefficients_[i][j], sfDer[i][0]);

        // get orientation part
        Dune::FieldMatrix<ctype,4,dim> qResult = orientationFEFunction_->evaluateDerivative(local);
        for (int i=0; i<4; i++)
            for (int j=0; j<dim; j++)
                result[3+i][j] = qResult[i][j];

        return result;
    }

    /** \brief Evaluate the derivative of the function, if you happen to know the function value (much faster!)
        \param local Local coordinates in the reference element where to evaluate the derivative
        \param q Value of the local gfe function at 'local'.  If you provide something wrong here the result will be wrong, too!
     */
    DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local,
                                                                                       const TargetSpace& q) const
    {
        Dune::FieldMatrix<ctype, embeddedDim, dim> result(0);

        // get translation part
        std::vector<Dune::FieldMatrix<ctype,1,dim> > sfDer(translationCoefficients_.size());
        localFiniteElement_.localBasis().evaluateJacobian(local, sfDer);

        for (size_t i=0; i<translationCoefficients_.size(); i++)
            for (int j=0; j<3; j++)
                result[j].axpy(translationCoefficients_[i][j], sfDer[i][0]);

        // get orientation part
        Dune::FieldMatrix<ctype,4,dim> qResult = orientationFEFunction_->evaluateDerivative(local,q.q);
        for (int i=0; i<4; i++)
            for (int j=0; j<dim; j++)
                result[3+i][j] = qResult[i][j];

        return result;
    }

    /** \brief For debugging: Evaluate the derivative of the function using a finite-difference approximation*/
    DerivativeType evaluateDerivativeFD(const Dune::FieldVector<ctype, dim>& local) const
    {
        Dune::FieldMatrix<ctype, embeddedDim, dim> result(0);

        // get translation part
        std::vector<Dune::FieldMatrix<ctype,1,dim> > sfDer(translationCoefficients_.size());
        localFiniteElement_.localBasis().evaluateJacobian(local, sfDer);

        for (size_t i=0; i<translationCoefficients_.size(); i++)
            for (int j=0; j<3; j++)
                result[j].axpy(translationCoefficients_[i][j], sfDer[i][0]);

        // get orientation part
        Dune::FieldMatrix<ctype,4,dim> qResult = orientationFEFunction_->evaluateDerivativeFD(local);
        for (int i=0; i<4; i++)
            for (int j=0; j<dim; j++)
                result[3+i][j] = qResult[i][j];

        return result;
    }

    /** \brief Evaluate the derivative of the function value with respect to a coefficient */
    void evaluateDerivativeOfValueWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                                 int coefficient,
                                                 Dune::FieldMatrix<double,embeddedDim,embeddedDim>& derivative) const
    {
        derivative = 0;

        // Translation part
        std::vector<Dune::FieldVector<ctype,1> > w;
        localFiniteElement_.localBasis().evaluateFunction(local,w);
        for (int i=0; i<3; i++)
            derivative[i][i] = w[coefficient];

        // Rotation part
        Dune::FieldMatrix<ctype,4,4> qDerivative;
        orientationFEFunction_->evaluateDerivativeOfValueWRTCoefficient(local,coefficient,qDerivative);
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                derivative[3+i][3+j] = qDerivative[i][j];
    }

    /** \brief Evaluate the derivative of the function value with respect to a coefficient */
    void evaluateFDDerivativeOfValueWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                                 int coefficient,
                                                 Dune::FieldMatrix<double,embeddedDim,embeddedDim>& derivative) const
    {
        derivative = 0;

        // Translation part
        std::vector<Dune::FieldVector<ctype,1> > w;
        localFiniteElement_.localBasis().evaluateFunction(local,w);
        for (int i=0; i<3; i++)
            derivative[i][i] = w[coefficient];

        // Rotation part
        Dune::FieldMatrix<ctype,4,4> qDerivative;
        orientationFEFunction_->evaluateFDDerivativeOfValueWRTCoefficient(local,coefficient,qDerivative);
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                derivative[3+i][3+j] = qDerivative[i][j];
    }

    /** \brief Evaluate the derivative of the gradient of the function with respect to a coefficient */
    void evaluateDerivativeOfGradientWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                                    int coefficient,
                                                    DerivativeOfGradientWRTCoefficientType& derivative) const
    {
        derivative = 0;

        // Translation part
        std::vector<Dune::FieldMatrix<ctype,1,dim> > w;
        localFiniteElement_.localBasis().evaluateJacobian(local,w);
        for (int i=0; i<3; i++)
            derivative[i][i] = w[coefficient][0];

        // Rotation part
        Tensor3<ctype,4,4,dim> qDerivative;
        orientationFEFunction_->evaluateDerivativeOfGradientWRTCoefficient(local,coefficient,qDerivative);
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                for (int k=0; k<dim; k++)
                    derivative[3+i][3+j][k] = qDerivative[i][j][k];
    }

    /** \brief Evaluate the derivative of the gradient of the function with respect to a coefficient */
    void evaluateFDDerivativeOfGradientWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                                    int coefficient,
                                                    DerivativeOfGradientWRTCoefficientType& derivative) const
    {
        derivative = 0;

        // Translation part
        std::vector<Dune::FieldMatrix<ctype,1,dim> > w;
        localFiniteElement_.localBasis().evaluateJacobian(local,w);
        for (int i=0; i<3; i++)
            derivative[i][i] = w[coefficient][0];

        // Rotation part
        Tensor3<ctype,4,4,dim> qDerivative;
        orientationFEFunction_->evaluateFDDerivativeOfGradientWRTCoefficient(local,coefficient,qDerivative);
        for (int i=0; i<4; i++)
            for (int j=0; j<4; j++)
                for (int k=0; k<dim; k++)
                    derivative[3+i][3+j][k] = qDerivative[i][j][k];
    }

    TargetSpace coefficient(int i) const {
        return TargetSpace(translationCoefficients_[i],orientationFEFunction_->coefficient(i));

    }
private:

    /** \brief The scalar local finite element, which provides the weighting factors
        \todo We really only need the local basis
     */
    const LocalFiniteElement& localFiniteElement_;

    // The two factors of a RigidBodyMotion
    //LocalGeodesicFEFunction<dim,ctype,RealTuple<3> > translationFEFunction_;

    std::vector<Dune::FieldVector<double,3> > translationCoefficients_;

    std::auto_ptr<LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,Rotation<double,3> > > orientationFEFunction_;
};


#endif
