#ifndef DUNE_GFE_FUNCTIONS_LOCALGEODESICFEFUNCTION_HH
#define DUNE_GFE_FUNCTIONS_LOCALGEODESICFEFUNCTION_HH

#include <vector>

#include <dune/common/fvector.hh>

#include <dune/geometry/type.hh>

#include <dune/gfe/averagedistanceassembler.hh>
#include <dune/gfe/targetspacertrsolver.hh>
#include <dune/gfe/functions/localquickanddirtyfefunction.hh>
#include <dune/gfe/spaces/productmanifold.hh>
#include <dune/gfe/spaces/realtuple.hh>
#include <dune/gfe/spaces/rotation.hh>

#include <dune/gfe/tensor3.hh>
#include <dune/gfe/tensorssd.hh>
#include <dune/gfe/linearalgebra.hh>


namespace Dune::GFE
{

// forward declaration
template <class LocalFiniteElement, class TargetSpace>
class LocalGfeTestFunctionBasis;

/** \brief A function defined by simplicial geodesic interpolation
           from the reference element to a Riemannian manifold.

   \tparam dim Dimension of the reference element
   \tparam ctype Type used for coordinates on the reference element
   \tparam LocalFiniteElement A Lagrangian finite element whose shape functions define the interpolation weights
   \tparam TS TargetSpace: The manifold that the function takes its values in
 */
template <int dim, class ctype, class LocalFiniteElement, class TS>
class LocalGeodesicFEFunction
{
public:
  using TargetSpace = TS;

private:
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

  /** \brief Rebind the FEFunction to another TargetSpace */
  template<class U>
  struct rebind
  {
    using other = LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,U>;
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

  /** \brief The scalar finite element used as the interpolation weights
   *
   * \note This method was added for InterpolationDerivatives, which needs it
   * to construct a copy of a LocalGeodesicFEFunction with ADOL-C's adouble
   * number type.  This is not optimal, because the localFiniteElement
   * really is an implementation detail of LocalGeodesicFEFunction and
   * should not be needed just to copy an entire object.  Other non-Euclidean
   * interpolation rules may not have such a finite element at all.
   * Therefore, this method may disappear again eventually.
   */
  const LocalFiniteElement& localFiniteElement() const
  {
    return localFiniteElement_;
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

  /** \brief Evaluate the value and the derivative of the interpolation function
   */
  std::pair<TargetSpace,DerivativeType> evaluateValueAndDerivative(const Dune::FieldVector<ctype, dim>& local) const;

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
  const TargetSpace& coefficient(int i) const
  {
    return coefficients_[i];
  }
private:

  static SymmetricMatrix<RT,embeddedDim> pseudoInverse(const SymmetricMatrix<RT,embeddedDim>& dFdq,
                                                             const TargetSpace& q)
  {
    const int shortDim = TargetSpace::TangentVector::dimension;

    // the orthonormal frame
    Dune::FieldMatrix<RT,shortDim,embeddedDim> O = q.orthonormalFrame();

    // compute A = O dFDq O^T
    // TODO A really is symmetric, too
    Dune::FieldMatrix<RT,shortDim,shortDim> A;
    for (int i=0; i<shortDim; i++)
      for (int j=0; j<shortDim; j++) {
        A[i][j] = 0;
        for (int k=0; k<embeddedDim; k++)
          for (int l=0; l<embeddedDim; l++)
            A[i][j] += O[i][k] * ((k>=l) ? dFdq(k,l) : dFdq(l,k)) *O[j][l];
      }

    A.invert();

    SymmetricMatrix<RT,embeddedDim> result;
    result = 0.0;
    for (int i=0; i<embeddedDim; i++)
      for (int j=0; j<=i; j++)
        for (int k=0; k<shortDim; k++)
          for (int l=0; l<shortDim; l++)
            result(i,j) += O[k][i]*A[k][l]*O[l][j];

    return result;
  }

  /** \brief Compute derivate of F(w,q) (the derivative of the weighted distance fctl) wrt to w */
  Dune::Matrix<RT> computeDFdw(const TargetSpace& q) const
  {
    Dune::Matrix<RT> dFdw(embeddedDim,localFiniteElement_.localBasis().size());
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

  // Create a reasonable initial iterate for the iterative solver
  Dune::GFE::LocalQuickAndDirtyFEFunction<dim,ctype,LocalFiniteElement,TargetSpace> localProjectedFEFunction(localFiniteElement_, coefficients_);
  TargetSpace initialIterate = localProjectedFEFunction.evaluate(local);

  // Iteratively solve the GFE minimization problem
  TargetSpaceRiemannianTRSolver<TargetSpace> solver;

  solver.setup(&assembler,
               initialIterate,
               1e-14,      // tolerance
               100         // maxNewtonSteps
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
  std::vector<Dune::FieldMatrix<ctype,1,dim> > B(coefficients_.size());
  localFiniteElement_.localBasis().evaluateJacobian(local, B);

  // compute negative derivative of F(w,q) (the derivative of the weighted distance fctl) wrt to w
  Dune::Matrix<RT> dFdw = computeDFdw(q);
  dFdw *= -1;

  // multiply the two previous matrices: the result is the right hand side
  // RHS = dFdw * B;
  // We need to write this multiplication by hand, because B is actually an (#coefficients) times 1 times dim matrix,
  // and we need to ignore the middle index.
  Dune::Matrix<RT> RHS(dFdw.N(), dim);

  for (size_t i=0; i<RHS.N(); i++)
    for (size_t j=0; j<RHS.M(); j++) {
      RHS[i][j] = 0;
      for (size_t k=0; k<dFdw.M(); k++)
        RHS[i][j] += dFdw[i][k]*B[k][0][j];
    }

  // the actual system matrix
  std::vector<Dune::FieldVector<ctype,1> > w;
  localFiniteElement_.localBasis().evaluateFunction(local, w);

  AverageDistanceAssembler<TargetSpace> assembler(coefficients_, w);

  SymmetricMatrix<RT,embeddedDim> dFdq;
  assembler.assembleEmbeddedHessian(q,dFdq);

  // We want to solve
  //     dFdq * x = rhs
  // We use the Gram-Schmidt solver because we know that dFdq is rank-deficient.

  const int shortDim = TargetSpace::TangentVector::dimension;

  // the orthonormal frame
  Dune::FieldMatrix<RT,shortDim,embeddedDim> basis = q.orthonormalFrame();
  GramSchmidtSolver<RT,shortDim,embeddedDim> gramSchmidtSolver(dFdq, basis);

  for (int i=0; i<dim; i++) {

    Dune::FieldVector<RT,embeddedDim> rhs;
    for (int j=0; j<embeddedDim; j++)
      rhs[j] = RHS[j][i];

    Dune::FieldVector<RT,embeddedDim> x;
    gramSchmidtSolver.solve(x, rhs);

    for (int j=0; j<embeddedDim; j++)
      result[j][i] = x[j];

  }

  return result;
}

template <int dim, class ctype, class LocalFiniteElement, class TargetSpace>
std::pair<TargetSpace,typename LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::DerivativeType>
LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,TargetSpace>::
evaluateValueAndDerivative(const Dune::FieldVector<ctype, dim>& local) const
{
  std::pair<TargetSpace,DerivativeType> result;
  result.first = evaluate(local);
  result.second = evaluateDerivative(local,result.first);
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

  SymmetricMatrix<RT,embeddedDim> dFdq;
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
          A[i][j] += O[i][k] * ((k>=l) ? dFdq(k,l) : dFdq(l,k)) * O[j][l];
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
  Dune::Matrix<RT> B(coefficients_.size(), dim);
  for (size_t i=0; i<coefficients_.size(); i++)
    for (size_t j=0; j<dim; j++)
      B[i][j] = BNested[i][0][j];

  // the actual system matrix
  std::vector<Dune::FieldVector<ctype,1> > w;
  localFiniteElement_.localBasis().evaluateFunction(local,w);

  AverageDistanceAssembler<TargetSpace> assembler(coefficients_, w);

  /** \todo Use a symmetric matrix here */
  SymmetricMatrix<RT,embeddedDim> dFdq;
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
  SymmetricMatrix<RT,embeddedDim> dFdqPseudoInv = pseudoInverse(dFdq,q);

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
    SymmetricMatrix<RT,embeddedDim> hesse = TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(coefficients_[k], q);
    for (int i=0; i<embeddedDim; i++)
      for (int j=0; j<=i; j++)
        dqdwF(i, j, k) = dqdwF(j, i, k) = hesse(i,j);

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
          // TODO Smarter implementation of the product with a symmetric matrix
          result[i][j][k] += ((j>=l) ? dFdqPseudoInv(j,l) : dFdqPseudoInv(l,j)) * foo[i][l][k];

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


/** \brief A function defined by geodesic interpolation
           from the reference element to a ProductManifold<RealTuple,Rotation>.

   This is a specialization for speeding up the code.

   \tparam dim Dimension of the reference element
   \tparam ctype Type used for coordinates on the reference element
 */
template <int dim, class ctype, class LocalFiniteElement, class field_type>
class LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement, Dune::GFE::ProductManifold<RealTuple<field_type,3>, Rotation<field_type,3> > >
{
  using TargetSpace = Dune::GFE::ProductManifold<RealTuple<field_type,3>, Rotation<field_type,3> >;

  typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
  static const int embeddedDim = EmbeddedTangentVector::dimension;

  static const int spaceDim = TargetSpace::TangentVector::dimension;

public:

  /** \brief The type used for derivatives */
  typedef Dune::FieldMatrix<field_type, embeddedDim, dim> DerivativeType;

  /** \brief The type used for derivatives of the gradient with respect to coefficients */
  typedef Tensor3<field_type,embeddedDim,embeddedDim,dim> DerivativeOfGradientWRTCoefficientType;

  /** \brief Constructor */
  LocalGeodesicFEFunction(const LocalFiniteElement& localFiniteElement,
                          const std::vector<TargetSpace>& coefficients)
    : localFiniteElement_(localFiniteElement),
    coefficients_(coefficients),
    translationCoefficients_(coefficients.size())
  {
    using namespace Dune::Indices;
    assert(localFiniteElement.localBasis().size() == coefficients.size());

    for (size_t i=0; i<coefficients.size(); i++)
      translationCoefficients_[i] = coefficients[i][_0].globalCoordinates();

    std::vector<Rotation<field_type,3> > orientationCoefficients(coefficients.size());
    for (size_t i=0; i<coefficients.size(); i++)
      orientationCoefficients[i] = coefficients[i][_1];

    orientationFEFunction_ = std::unique_ptr<LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,Rotation<field_type,3> > > (new LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,Rotation<field_type,3> >(localFiniteElement,orientationCoefficients));

  }

  /** \brief Rebind the FEFunction to another TargetSpace */
  template<class U>
  struct rebind
  {
    using other = LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,U>;
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
  TargetSpace evaluate(const Dune::FieldVector<ctype, dim>& local) const
  {
    using namespace Dune::Indices;

    TargetSpace result;

    // Evaluate the weighting factors---these are the Lagrangian shape function values at 'local'
    std::vector<Dune::FieldVector<ctype,1> > w;
    localFiniteElement_.localBasis().evaluateFunction(local,w);

    result[_0] = Dune::FieldVector<field_type,3>(0.0);
    for (size_t i=0; i<w.size(); i++)
      result[_0].globalCoordinates().axpy(w[i][0], translationCoefficients_[i]);

    result[_1] = orientationFEFunction_->evaluate(local);
    return result;
  }

  /** \brief Evaluate the derivative of the function */
  DerivativeType evaluateDerivative(const Dune::FieldVector<ctype, dim>& local) const
  {
    DerivativeType result(0);

    // get translation part
    std::vector<Dune::FieldMatrix<ctype,1,dim> > sfDer(translationCoefficients_.size());
    localFiniteElement_.localBasis().evaluateJacobian(local, sfDer);

    for (size_t i=0; i<translationCoefficients_.size(); i++)
      for (int j=0; j<3; j++)
        result[j].axpy(translationCoefficients_[i][j], sfDer[i][0]);

    // get orientation part
    Dune::FieldMatrix<field_type,4,dim> qResult = orientationFEFunction_->evaluateDerivative(local);
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
    using namespace Dune::Indices;

    DerivativeType result(0);

    // get translation part
    std::vector<Dune::FieldMatrix<ctype,1,dim> > sfDer(translationCoefficients_.size());
    localFiniteElement_.localBasis().evaluateJacobian(local, sfDer);

    for (size_t i=0; i<translationCoefficients_.size(); i++)
      for (int j=0; j<3; j++)
        result[j].axpy(translationCoefficients_[i][j], sfDer[i][0]);

    // get orientation part
    Dune::FieldMatrix<field_type,4,dim> qResult = orientationFEFunction_->evaluateDerivative(local,q[_1]);
    for (int i=0; i<4; i++)
      for (int j=0; j<dim; j++)
        result[3+i][j] = qResult[i][j];

    return result;
  }

  /** \brief Evaluate the value and the derivative of the interpolation function
   */
  std::pair<TargetSpace,DerivativeType> evaluateValueAndDerivative(const Dune::FieldVector<ctype, dim>& local) const
  {
    std::pair<TargetSpace,DerivativeType> result;
    result.first = evaluate(local);
    result.second = evaluateDerivative(local,result.first);
    return result;
  }

  /** \brief Evaluate the derivative of the function value with respect to a coefficient */
  void evaluateDerivativeOfValueWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                               int coefficient,
                                               Dune::FieldMatrix<field_type,embeddedDim,embeddedDim>& derivative) const
  {
    derivative = 0;

    // Translation part
    std::vector<Dune::FieldVector<ctype,1> > w;
    localFiniteElement_.localBasis().evaluateFunction(local,w);
    for (int i=0; i<3; i++)
      derivative[i][i] = w[coefficient];

    // Rotation part
    Dune::FieldMatrix<field_type,4,4> qDerivative;
    orientationFEFunction_->evaluateDerivativeOfValueWRTCoefficient(local,coefficient,qDerivative);
    for (int i=0; i<4; i++)
      for (int j=0; j<4; j++)
        derivative[3+i][3+j] = qDerivative[i][j];
  }

  /** \brief Evaluate the derivative of the function value with respect to a coefficient */
  void evaluateFDDerivativeOfValueWRTCoefficient(const Dune::FieldVector<ctype, dim>& local,
                                                 int coefficient,
                                                 Dune::FieldMatrix<field_type,embeddedDim,embeddedDim>& derivative) const
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
    derivative = field_type(0);

    // Translation part
    std::vector<Dune::FieldMatrix<ctype,1,dim> > w;
    localFiniteElement_.localBasis().evaluateJacobian(local,w);
    for (int i=0; i<3; i++)
      derivative[i][i] = w[coefficient][0];

    // Rotation part
    Tensor3<field_type,4,4,dim> qDerivative;
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

  const TargetSpace& coefficient(int i) const {
    return coefficients_[i];
  }

private:

  /** \brief The scalar local finite element, which provides the weighting factors
      \todo We really only need the local basis
   */
  const LocalFiniteElement& localFiniteElement_;

  // The coefficients of this interpolation rule
  std::vector<TargetSpace> coefficients_;

  // The coefficients again.  Yes, we store it twice:  To evaluate the interpolation rule efficiently
  // we need access to the coefficients for the various factor spaces separately.
  std::vector<Dune::FieldVector<field_type,3> > translationCoefficients_;

  std::unique_ptr<LocalGeodesicFEFunction<dim,ctype,LocalFiniteElement,Rotation<field_type,3> > > orientationFEFunction_;
};

}  // namespace Dune::GFE

#endif
