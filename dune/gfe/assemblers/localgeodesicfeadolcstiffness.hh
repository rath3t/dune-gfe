#ifndef DUNE_GFE_LOCAL_GEODESIC_FE_ADOL_C_STIFFNESS_HH
#define DUNE_GFE_LOCAL_GEODESIC_FE_ADOL_C_STIFFNESS_HH

#include <adolc/adouble.h>
#include <adolc/taping.h>
#include <adolc/drivers/drivers.h>

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>

#include <dune/gfe/assemblers/localenergy.hh>
#include <dune/gfe/assemblers/localgeodesicfestiffness.hh>
#include <dune/gfe/spaces/productmanifold.hh>

/** \brief Assembles energy gradient and Hessian with ADOL-C (automatic differentiation)
 */
template<class Basis, class TargetSpace>
class LocalGeodesicFEADOLCStiffness
  : public LocalGeodesicFEStiffness<Basis,TargetSpace>
{
  // grid types
  typedef typename Basis::GridView GridView;
  typedef typename GridView::ctype DT;
  typedef typename TargetSpace::ctype RT;
  constexpr static int gridDim = GridView::dimension;

  // The 'active' target spaces, i.e., the number type is replaced by adouble
  using ATargetSpace = typename TargetSpace::template rebind<adouble>::other;


public:

  //! Dimension of a tangent space
  constexpr static int blocksize = TargetSpace::TangentVector::dimension;

  LocalGeodesicFEADOLCStiffness(const Dune::GFE::LocalEnergy<Basis, ATargetSpace>* energy, bool adolcScalarMode = false)
    : localEnergy_(energy),
    adolcScalarMode_(adolcScalarMode)
  {}

  /** \brief Compute the energy at the current configuration */
  virtual RT energy(const typename Basis::LocalView& localView,
                    const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Coefficients& coefficients) const override;

  /** \brief ProductManifolds: Compute the energy from coefficients in separate containers
   * for each factor
   */
  virtual RT energy (const typename Basis::LocalView& localView,
                     const typename Dune::GFE::Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& coefficients) const override;

  /** \brief Assemble the element gradient of the energy functional
   */
  virtual void assembleGradient(const typename Basis::LocalView& localView,
                                const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Coefficients& coefficients,
                                typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Gradient& gradient) const override;

  /** \brief Assemble the local stiffness matrix at the current position

     This uses the automatic differentiation toolbox ADOL_C.
   */
  virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                          const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Coefficients& coefficients,
                                          typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Gradient& localGradient,
                                          typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Hessian& localHessian) const override;

  /** \brief Assemble the local stiffness matrix at the current position
   */
  virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                          const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeCoefficients& coefficients,
                                          typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeGradient& localGradient,
                                          typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeHessian& localHessian) const override;

  const Dune::GFE::LocalEnergy<Basis, ATargetSpace>* localEnergy_;
  const bool adolcScalarMode_;
};


template <class Basis, class TargetSpace>
typename LocalGeodesicFEADOLCStiffness<Basis, TargetSpace>::RT
LocalGeodesicFEADOLCStiffness<Basis, TargetSpace>::
energy(const typename Basis::LocalView& localView,
       const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Coefficients& localSolution) const
{
  double pureEnergy;
  int rank = localView.globalBasis().gridView().comm().rank();
  std::vector<ATargetSpace> localASolution(localSolution.size());

  trace_on(rank);

  adouble energy = 0;

  // The following loop is not quite intuitive: we copy the localSolution into an
  // array of FieldVector<double>, go from there to FieldVector<adouble> and
  // only then to ATargetSpace.
  // Rationale: The constructor/assignment-from-vector of TargetSpace frequently
  // contains a projection onto the manifold from the surrounding Euclidean space.
  // ADOL-C needs a function on the whole Euclidean space, hence that projection
  // is part of the function and needs to be taped.

  // The following variable cannot be declared inside of the loop, or ADOL-C will report wrong results
  // (Presumably because several independent variables use the same memory location.)
  std::vector<typename ATargetSpace::CoordinateType> aRaw(localSolution.size());
  for (size_t i=0; i<localSolution.size(); i++) {
    typename TargetSpace::CoordinateType raw = localSolution[i].globalCoordinates();
    for (size_t j=0; j<raw.size(); j++)
      aRaw[i][j] <<= raw[j];
    localASolution[i] = aRaw[i];    // may contain a projection onto M -- needs to be done in adouble
  }

  try {
    energy = localEnergy_->energy(localView,localASolution);
  } catch (Dune::Exception &e) {
    trace_off();
    throw e;
  }

  energy >>= pureEnergy;

  trace_off();

  return pureEnergy;
}


template <class Basis, class TargetSpace>
typename LocalGeodesicFEADOLCStiffness<Basis, TargetSpace>::RT
LocalGeodesicFEADOLCStiffness<Basis, TargetSpace>::
energy(const typename Basis::LocalView& localView,
       const typename Dune::GFE::Impl::LocalEnergyTypes<TargetSpace>::CompositeCoefficients& localConfiguration) const
{
  using namespace Dune::Indices;

  int rank = Dune::MPIHelper::getCommunication().rank();

  if constexpr (not Dune::GFE::Impl::LocalEnergyTypes<TargetSpace>::isProductManifold)
  {
    // TODO: The following fails in the clang-11 C++20 CI jobs
    //static_assert(localConfiguration.size()==1);
    return energy(localView, localConfiguration[_0]);
  }
  else
  {
    using TargetSpace0 = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_0])>;
    using TargetSpace1 = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_1])>;

    using ATargetSpace0 = typename TargetSpace0::template rebind<adouble>::other;
    using ATargetSpace1 = typename TargetSpace1::template rebind<adouble>::other;

    Dune::TupleVector<std::vector<ATargetSpace0>, std::vector<ATargetSpace1> > localAConfiguration;
    localAConfiguration[_0].resize(localConfiguration[_0].size());
    localAConfiguration[_1].resize(localConfiguration[_1].size());

    double pureEnergy;
    trace_on(rank);

    adouble energy = 0;

    // The following loop is not quite intuitive: we copy the localSolution into an
    // array of FieldVector<double>, go from there to FieldVector<adouble> and
    // only then to ATargetSpace.
    // Rationale: The constructor/assignment-from-vector of TargetSpace frequently
    // contains a projection onto the manifold from the surrounding Euclidean space.
    // ADOL-C needs a function on the whole Euclidean space, hence that projection
    // is part of the function and needs to be taped.

    // The following variable cannot be declared inside of the loop, or ADOL-C will report wrong results
    // (Presumably because several independent variables use the same memory location.)
    std::vector<typename ATargetSpace0::CoordinateType> aRaw0(localConfiguration[_0].size());
    for (size_t i=0; i<localConfiguration[_0].size(); i++) {
      typename TargetSpace0::CoordinateType raw = localConfiguration[_0][i].globalCoordinates();
      for (size_t j=0; j<raw.size(); j++)
        aRaw0[i][j] <<= raw[j];
      localAConfiguration[_0][i] = aRaw0[i];    // may contain a projection onto M -- needs to be done in adouble
    }

    std::vector<typename ATargetSpace1::CoordinateType> aRaw1(localConfiguration[_1].size());
    for (size_t i=0; i<localConfiguration[_1].size(); i++) {
      typename TargetSpace1::CoordinateType raw = localConfiguration[_1][i].globalCoordinates();
      for (size_t j=0; j<raw.size(); j++)
        aRaw1[i][j] <<= raw[j];
      localAConfiguration[_1][i] = aRaw1[i];    // may contain a projection onto M -- needs to be done in adouble
    }

    try {

      energy = localEnergy_->energy(localView, localAConfiguration);

    } catch (Dune::Exception &e) {
      trace_off();
      throw e;
    }
    energy >>= pureEnergy;

    trace_off();

    return pureEnergy;
  }
}


template <class Basis, class TargetSpace>
void LocalGeodesicFEADOLCStiffness<Basis, TargetSpace>::
assembleGradient(const typename Basis::LocalView& localView,
                 const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Coefficients& localSolution,
                 typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Gradient& localGradient) const
{
  // Tape energy computation.  We may not have to do this every time, but it's comparatively cheap.
  energy(localView, localSolution);

  int rank = localView.globalBasis().gridView().comm().rank();

  // Compute the actual gradient
  constexpr static auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;
  size_t nDofs = localSolution.size();
  size_t nDoubles = nDofs*embeddedBlocksize;
  std::vector<double> xp(nDoubles);
  int idx=0;
  for (size_t i=0; i<nDofs; i++)
    for (size_t j=0; j<embeddedBlocksize; j++)
      xp[idx++] = localSolution[i].globalCoordinates()[j];

  // Compute gradient
  std::vector<double> g(nDoubles);
  gradient(rank,nDoubles,xp.data(),g.data());                    // gradient evaluation

  // Copy into Dune type
  std::vector<typename TargetSpace::EmbeddedTangentVector> localEmbeddedGradient(localSolution.size());

  idx=0;
  for (size_t i=0; i<nDofs; i++)
    for (size_t j=0; j<embeddedBlocksize; j++)
      localEmbeddedGradient[i][j] = g[idx++];

  // Express gradient in local coordinate system
  for (size_t i=0; i<nDofs; i++) {
    Dune::FieldMatrix<RT,blocksize,embeddedBlocksize> orthonormalFrame = localSolution[i].orthonormalFrame();
    orthonormalFrame.mv(localEmbeddedGradient[i],localGradient[i]);
  }
}


// ///////////////////////////////////////////////////////////
//   Compute gradient and Hessian together
//   To compute the Hessian we need to compute the gradient anyway, so we may
//   as well return it.  This saves assembly time.
// ///////////////////////////////////////////////////////////
template <class Basis, class TargetSpace>
void LocalGeodesicFEADOLCStiffness<Basis, TargetSpace>::
assembleGradientAndHessian(const typename Basis::LocalView& localView,
                           const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Coefficients& localSolution,
                           typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Gradient& localGradient,
                           typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::Hessian& localHessian) const
{
  // Tape energy computation.  We may not have to do this every time, but it's comparatively cheap.
  energy(localView, localSolution);

  int rank = localView.globalBasis().gridView().comm().rank();

  /////////////////////////////////////////////////////////////////
  // Compute the gradient.  It is needed to transform the Hessian
  // into the correct coordinates.
  /////////////////////////////////////////////////////////////////

  // Compute the actual gradient
  constexpr static auto embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension;
  size_t nDofs = localSolution.size();
  size_t nDoubles = nDofs*embeddedBlocksize;
  std::vector<double> xp(nDoubles);
  int idx=0;
  for (size_t i=0; i<nDofs; i++)
    for (size_t j=0; j<embeddedBlocksize; j++)
      xp[idx++] = localSolution[i].globalCoordinates()[j];

  // Compute gradient
  std::vector<double> g(nDoubles);
  gradient(rank,nDoubles,xp.data(),g.data());                    // gradient evaluation

  // Copy into Dune type
  std::vector<typename TargetSpace::EmbeddedTangentVector> localEmbeddedGradient(localSolution.size());

  idx=0;
  for (size_t i=0; i<nDofs; i++)
    for (size_t j=0; j<embeddedBlocksize; j++)
      localEmbeddedGradient[i][j] = g[idx++];

  // Express gradient in local coordinate system
  for (size_t i=0; i<nDofs; i++) {
    Dune::FieldMatrix<RT,blocksize,embeddedBlocksize> orthonormalFrame = localSolution[i].orthonormalFrame();
    orthonormalFrame.mv(localEmbeddedGradient[i],localGradient[i]);
  }

  /////////////////////////////////////////////////////////////////
  // Compute Hessian
  /////////////////////////////////////////////////////////////////

  // We compute the Hessian of the energy functional using the ADOL-C system.
  // Since ADOL-C does not know about nonlinear spaces, what we get is actually
  // the Hessian of a prolongation of the energy functional into the surrounding
  // Euclidean space.  To obtain the Riemannian Hessian from this we apply the
  // formula described in Absil, Mahoney, Trumpf, "An extrinsic look at the Riemannian Hessian".
  // This formula consists of two steps:
  // 1) Remove all entries of the Hessian pertaining to the normal space of the
  //    manifold.  In the aforementioned paper this is done by projection onto the
  //    tangent space.  Since we want a matrix that is really smaller (but full rank again),
  //    we can achieve the same effect by multiplying the embedded Hessian from the left
  //    and from the right by the matrix of orthonormal frames.
  // 2) Add a correction involving the Weingarten map.
  //
  // This works, and is easy to implement using the ADOL-C "hessian" driver.
  // However, here we implement a small shortcut.  Computing the embedded Hessian and
  // multiplying one side by the orthonormal frame is the same as evaluating the Hessian
  // (seen as an operator from R^n to R^n) in the directions of the vectors of the
  // orthonormal frame.  By luck, ADOL-C can compute the evaluations of the Hessian in
  // a given direction directly (in fact, this is also how the "hessian" driver works).
  // Since there are less frame vectors than the dimension of the embedding space,
  // this reinterpretation allows to reduce the number of calls to ADOL-C.
  // In my Cosserat shell tests this reduced assembly time by about 10%.

  std::vector<Dune::FieldMatrix<RT,blocksize,embeddedBlocksize> > orthonormalFrame(nDofs);

  for (size_t i=0; i<nDofs; i++)
    orthonormalFrame[i] = localSolution[i].orthonormalFrame();

  Dune::Matrix<Dune::FieldMatrix<double,blocksize, embeddedBlocksize> > embeddedHessian(nDofs,nDofs);

  if (adolcScalarMode_) {
    std::vector<double> v(nDoubles);
    std::vector<double> w(nDoubles);

    std::fill(v.begin(), v.end(), 0.0);

    for (size_t i=0; i<nDofs; i++)
      for (int ii=0; ii<blocksize; ii++)
      {
        // Evaluate Hessian in the direction of each vector of the orthonormal frame
        for (size_t k=0; k<embeddedBlocksize; k++)
          v[i*embeddedBlocksize + k] = orthonormalFrame[i][ii][k];

        int rc= 3;
        MINDEC(rc, hess_vec(rank, nDoubles, xp.data(), v.data(), w.data()));
        if (rc < 0)
          DUNE_THROW(Dune::Exception, "ADOL-C has returned with error code " << rc << "!");

        for (size_t j=0; j<nDoubles; j++)
          embeddedHessian[i][j/embeddedBlocksize][ii][j%embeddedBlocksize] = w[j];

        // Make v the null vector again
        std::fill(&v[i*embeddedBlocksize], &v[(i+1)*embeddedBlocksize], 0.0);
      }
  } else {   //ADOL-C vector mode
    int n = nDoubles;
    int nDirections = nDofs * blocksize;
    double* tangent[nDoubles];
    for(size_t i=0; i<nDoubles; i++)
      tangent[i] = (double*)malloc(nDirections*sizeof(double));

    double* rawHessian[nDoubles];
    for(size_t i=0; i<nDoubles; i++)
      rawHessian[i] = (double*)malloc(nDirections*sizeof(double));

    for (int j=0; j<nDirections; j++)
    {
      for (int i=0; i<n; i++)
        tangent[i][j] = 0.0;

      for (int i=0; i<embeddedBlocksize; i++)
        tangent[(j/blocksize)*embeddedBlocksize+i][j] = orthonormalFrame[j/blocksize][j%blocksize][i];
    }
    hess_mat(rank,nDoubles,nDirections,xp.data(),tangent,rawHessian);

    // Copy Hessian into Dune data type
    for(size_t i=0; i<nDoubles; i++)
      for (int j=0; j<nDirections; j++)
        embeddedHessian[j/blocksize][i/embeddedBlocksize][j%blocksize][i%embeddedBlocksize] = rawHessian[i][j];

    for(size_t i=0; i<nDoubles; i++) {
      free(rawHessian[i]);
      free(tangent[i]);
    }
  }
  // From this, compute the Hessian with respect to the manifold (which we assume here is embedded
  // isometrically in a Euclidean space.
  // For the detailed explanation of the following see: Absil, Mahoney, Trumpf, "An extrinsic look
  // at the Riemannian Hessian".

  typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
  typedef typename TargetSpace::TangentVector TangentVector;

  localHessian.setSize(nDofs,nDofs);

  for (size_t col=0; col<nDofs; col++) {

    for (size_t subCol=0; subCol<blocksize; subCol++) {

      EmbeddedTangentVector z = orthonormalFrame[col][subCol];

      // P_x \partial^2 f z
      for (size_t row=0; row<nDofs; row++) {
        TangentVector semiEmbeddedProduct;
        embeddedHessian[row][col].mv(z,semiEmbeddedProduct);

        for (int subRow=0; subRow<blocksize; subRow++)
          localHessian[row][col][subRow][subCol] = semiEmbeddedProduct[subRow];
      }

    }

  }

  //////////////////////////////////////////////////////////////////////////////////////
  //  Further correction due to non-planar configuration space
  //  + \mathfrak{A}_x(z,P^\orth_x \partial f)
  //////////////////////////////////////////////////////////////////////////////////////

  // Project embedded gradient onto normal space
  std::vector<typename TargetSpace::EmbeddedTangentVector> projectedGradient(localSolution.size());
  for (size_t i=0; i<localSolution.size(); i++)
    projectedGradient[i] = localSolution[i].projectOntoNormalSpace(localEmbeddedGradient[i]);

  for (size_t row=0; row<nDofs; row++) {

    for (size_t subRow=0; subRow<blocksize; subRow++) {

      EmbeddedTangentVector z = orthonormalFrame[row][subRow];
      EmbeddedTangentVector tmp1 = localSolution[row].weingarten(z,projectedGradient[row]);

      TangentVector tmp2;
      orthonormalFrame[row].mv(tmp1,tmp2);

      localHessian[row][row][subRow] += tmp2;
    }

  }
}


/////////////////////////////////////////////////////////////////////////////////////
//   Compute gradient and Hessian together
//   To compute the Hessian we need to compute the gradient anyway, so we may
//   as well return it.  This saves assembly time.
/////////////////////////////////////////////////////////////////////////////////////
template <class Basis, class TargetSpace>
void LocalGeodesicFEADOLCStiffness<Basis, TargetSpace>::
assembleGradientAndHessian(const typename Basis::LocalView& localView,
                           const typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeCoefficients& localConfiguration,
                           typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeGradient& localGradient,
                           typename Dune::GFE::Impl::LocalStiffnessTypes<TargetSpace>::CompositeHessian& localHessian) const
{
  using namespace Dune::Indices;

  if constexpr (not Dune::GFE::Impl::LocalEnergyTypes<TargetSpace>::isProductManifold)
  {
    // TODO: The following fails in the clang-11 C++20 CI jobs
    //static_assert(localConfiguration.size()==1);
    return assembleGradientAndHessian(localView,
                                      localConfiguration[_0],
                                      localGradient[_0],
                                      localHessian[_0][_0]);
  }
  else
  {
    int rank = Dune::MPIHelper::getCommunication().rank();

    // Tape energy computation.  We may not have to do this every time, but it's comparatively cheap.
    energy(localView, localConfiguration);

    /////////////////////////////////////////////////////////////////
    // Compute the gradient.  It is needed to transform the Hessian
    // into the correct coordinates.
    /////////////////////////////////////////////////////////////////

    // TODO: The following fails in the clang-11 C++20 CI jobs
    //static_assert(localConfiguration.size()==2, "ADOL-C assembly only implemented for product spaces with two factors");

    //! Dimension of the embedding space
    using TargetSpace0 = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_0])>;
    using TargetSpace1 = std::decay_t<decltype(std::declval<TargetSpace>()[Dune::Indices::_1])>;

    constexpr static int blocksize0 = TargetSpace0::TangentVector::dimension;
    constexpr static int blocksize1 = TargetSpace1::TangentVector::dimension;

    constexpr static int embeddedBlocksize0 = TargetSpace0::EmbeddedTangentVector::dimension;
    constexpr static int embeddedBlocksize1 = TargetSpace1::EmbeddedTangentVector::dimension;

    // Compute the actual gradient
    size_t nDofs0 = localConfiguration[_0].size();
    size_t nDofs1 = localConfiguration[_1].size();
    size_t nDoubles = nDofs0*embeddedBlocksize0 + nDofs1*embeddedBlocksize1;

    std::vector<double> xp(nDoubles);
    int idx=0;
    for (size_t i=0; i<localConfiguration[_0].size(); i++)
      for (size_t j=0; j<embeddedBlocksize0; j++)
        xp[idx++] = localConfiguration[_0][i].globalCoordinates()[j];

    for (size_t i=0; i<localConfiguration[_1].size(); i++)
      for (size_t j=0; j<embeddedBlocksize1; j++)
        xp[idx++] = localConfiguration[_1][i].globalCoordinates()[j];

    // Compute gradient
    std::vector<double> g(nDoubles);
    gradient(rank,nDoubles,xp.data(),g.data());

    // Copy into Dune type
    std::vector<typename TargetSpace0::EmbeddedTangentVector> localEmbeddedGradient0(localConfiguration[_0].size());
    std::vector<typename TargetSpace1::EmbeddedTangentVector> localEmbeddedGradient1(localConfiguration[_1].size());

    idx=0;
    for (size_t i=0; i<localConfiguration[_0].size(); i++) {
      for (size_t j=0; j<embeddedBlocksize0; j++)
        localEmbeddedGradient0[i][j] = g[idx++];

      // Express gradient in local coordinate system
      localConfiguration[_0][i].orthonormalFrame().mv(localEmbeddedGradient0[i],localGradient[_0][i]);
    }

    for (size_t i=0; i<localConfiguration[_1].size(); i++) {
      for (size_t j=0; j<embeddedBlocksize1; j++)
        localEmbeddedGradient1[i][j] = g[idx++];

      // Express gradient in local coordinate system
      localConfiguration[_1][i].orthonormalFrame().mv(localEmbeddedGradient1[i],localGradient[_1][i]);
    }

    /////////////////////////////////////////////////////////////////
    // Compute Hessian
    /////////////////////////////////////////////////////////////////

    // We compute the Hessian of the energy functional using the ADOL-C system.
    // Since ADOL-C does not know about nonlinear spaces, what we get is actually
    // the Hessian of a prolongation of the energy functional into the surrounding
    // Euclidean space.  To obtain the Riemannian Hessian from this we apply the
    // formula described in Absil, Mahoney, Trumpf, "An extrinsic look at the Riemannian Hessian".
    // This formula consists of two steps:
    // 1) Remove all entries of the Hessian pertaining to the normal space of the
    //    manifold.  In the aforementioned paper this is done by projection onto the
    //    tangent space.  Since we want a matrix that is really smaller (but full rank again),
    //    we can achieve the same effect by multiplying the embedded Hessian from the left
    //    and from the right by the matrix of orthonormal frames.
    // 2) Add a correction involving the Weingarten map.
    //
    // This works, and is easy to implement using the ADOL-C "hessian" driver.
    // However, here we implement a small shortcut.  Computing the embedded Hessian and
    // multiplying one side by the orthonormal frame is the same as evaluating the Hessian
    // (seen as an operator from R^n to R^n) in the directions of the vectors of the
    // orthonormal frame.  By luck, ADOL-C can compute the evaluations of the Hessian in
    // a given direction directly (in fact, this is also how the "hessian" driver works).
    // Since there are less frame vectors than the dimension of the embedding space,
    // this reinterpretation allows to reduce the number of calls to ADOL-C.
    // In my Cosserat shell tests this reduced assembly time by about 10%.

    std::vector<Dune::FieldMatrix<RT,blocksize0,embeddedBlocksize0> > orthonormalFrame0(nDofs0);

    for (size_t i=0; i<nDofs0; i++)
      orthonormalFrame0[i] = localConfiguration[_0][i].orthonormalFrame();

    std::vector<Dune::FieldMatrix<RT,blocksize1,embeddedBlocksize1> > orthonormalFrame1(nDofs1);

    for (size_t i=0; i<nDofs1; i++)
      orthonormalFrame1[i] = localConfiguration[_1][i].orthonormalFrame();

    Dune::Matrix<Dune::FieldMatrix<double,blocksize0, embeddedBlocksize0> > embeddedHessian00(nDofs0,nDofs0);
    Dune::Matrix<Dune::FieldMatrix<double,blocksize0, embeddedBlocksize1> > embeddedHessian01(nDofs0,nDofs1);
    Dune::Matrix<Dune::FieldMatrix<double,blocksize1, embeddedBlocksize0> > embeddedHessian10(nDofs1,nDofs0);
    Dune::Matrix<Dune::FieldMatrix<double,blocksize1, embeddedBlocksize1> > embeddedHessian11(nDofs1,nDofs1);

    if(adolcScalarMode_) {
      std::vector<double> v(nDoubles);
      std::vector<double> w(nDoubles);

      std::fill(v.begin(), v.end(), 0.0);

      size_t nDoubles0 = nDofs0*embeddedBlocksize0;   // nDoubles = nDoubles0 + nDoubles1
      size_t nDoubles1 = nDofs1*embeddedBlocksize1;

      std::fill(v.begin(), v.end(), 0.0);

      for (size_t i=0; i<nDofs0 + nDofs1; i++) {

        // Evaluate Hessian in the direction of each vector of the orthonormal frame
        if (i < nDofs0) {     //Upper half
          auto i0 = i;
          for (int ii0=0; ii0<blocksize0; ii0++) {
            for (size_t k0=0; k0<embeddedBlocksize0; k0++) {
              v[i0*embeddedBlocksize0 + k0] = orthonormalFrame0[i0][ii0][k0];
            }
            int rc= 3;
            MINDEC(rc, hess_vec(rank, nDoubles, xp.data(), v.data(), w.data()));
            if (rc < 0)
              DUNE_THROW(Dune::Exception, "ADOL-C has returned with error code " << rc << "!");

            for (size_t j0=0; j0<nDoubles0; j0++)         //Upper left
              embeddedHessian00[i0][j0/embeddedBlocksize0][ii0][j0%embeddedBlocksize0] = w[j0];

            for (size_t j1=0; j1<nDoubles1; j1++)         //Upper right
              embeddedHessian01[i0][j1/embeddedBlocksize1][ii0][j1%embeddedBlocksize1] = w[nDoubles0 + j1];

            // Make v the null vector again
            std::fill(&v[i0*embeddedBlocksize0], &v[(i0+1)*embeddedBlocksize0], 0.0);
          }
        } else  {     // i = nDofs0 ... nDofs0 + nDofs1 //lower half
          auto i1 = i - nDofs0;
          for (int ii1=0; ii1<blocksize1; ii1++) {
            for (size_t k1=0; k1<embeddedBlocksize1; k1++) {
              v[nDoubles0 + i1*embeddedBlocksize1 + k1] = orthonormalFrame1[i1][ii1][k1];
            }
            int rc= 3;
            MINDEC(rc, hess_vec(rank, nDoubles, xp.data(), v.data(), w.data()));
            if (rc < 0)
              DUNE_THROW(Dune::Exception, "ADOL-C has returned with error code " << rc << "!");

            for (size_t j0=0; j0<nDoubles0; j0++)         // Upper left
              embeddedHessian10[i1][j0/embeddedBlocksize0][ii1][j0%embeddedBlocksize0] = w[j0];

            for (size_t j1=0; j1<nDoubles1; j1++)         //Upper right
              embeddedHessian11[i1][j1/embeddedBlocksize1][ii1][j1%embeddedBlocksize1] = w[nDoubles0 + j1];

            // Make v the null vector again
            std::fill(&v[nDoubles0 + i1*embeddedBlocksize1], &v[nDoubles0 + (i1+1)*embeddedBlocksize1], 0.0);
          }
        }
      }

    } else { //ADOL-C vector mode}
      int n = nDoubles;
      int nDirections = nDofs0 * blocksize0 + nDofs1 * blocksize1;
      double* tangent[nDoubles];
      for(size_t i=0; i<nDoubles; i++)
        tangent[i] = (double*)malloc(nDirections*sizeof(double));

      double* rawHessian[nDoubles];
      for(size_t i=0; i<nDoubles; i++)
        rawHessian[i] = (double*)malloc(nDirections*sizeof(double));

      // Initialize directions field with zeros
      for (int j=0; j<nDirections; j++)
        for (int i=0; i<n; i++)
          tangent[i][j] = 0.0;

      for (size_t j=0; j<nDofs0*blocksize0; j++)
        for (int i=0; i<embeddedBlocksize0; i++)
          tangent[(j/blocksize0)*embeddedBlocksize0+i][j] = orthonormalFrame0[j/blocksize0][j%blocksize0][i];

      for (size_t j=0; j<nDofs1*blocksize1; j++)
        for (int i=0; i<embeddedBlocksize1; i++)
          tangent[nDofs0*embeddedBlocksize0 + (j/blocksize1)*embeddedBlocksize1+i][nDofs0*blocksize0 + j] = orthonormalFrame1[j/blocksize1][j%blocksize1][i];

      hess_mat(rank,nDoubles,nDirections,xp.data(),tangent,rawHessian);

      // Copy Hessian into Dune data type
      size_t offset0 = nDofs0*embeddedBlocksize0;
      size_t offset1 = nDofs0*blocksize0;

      // upper left block
      for(size_t i=0; i<nDofs0*embeddedBlocksize0; i++)
        for (size_t j=0; j<nDofs0*blocksize0; j++)
          embeddedHessian00[j/blocksize0][i/embeddedBlocksize0][j%blocksize0][i%embeddedBlocksize0] = rawHessian[i][j];

      // upper right block
      for(size_t i=0; i<nDofs1*embeddedBlocksize1; i++)
        for (size_t j=0; j<nDofs0*blocksize0; j++)
          embeddedHessian01[j/blocksize0][i/embeddedBlocksize1][j%blocksize0][i%embeddedBlocksize1] = rawHessian[offset0+i][j];

      // lower left block
      for(size_t i=0; i<nDofs0*embeddedBlocksize0; i++)
        for (size_t j=0; j<nDofs1*blocksize1; j++)
          embeddedHessian10[j/blocksize1][i/embeddedBlocksize0][j%blocksize1][i%embeddedBlocksize0] = rawHessian[i][offset1+j];

      // lower right block
      for(size_t i=0; i<nDofs1*embeddedBlocksize1; i++)
        for (size_t j=0; j<nDofs1*blocksize1; j++)
          embeddedHessian11[j/blocksize1][i/embeddedBlocksize1][j%blocksize1][i%embeddedBlocksize1] = rawHessian[offset0+i][offset1+j];

      for(size_t i=0; i<nDoubles; i++) {
        free(rawHessian[i]);
        free(tangent[i]);
      }
    }

    // From this, compute the Hessian with respect to the manifold (which we assume here is embedded
    // isometrically in a Euclidean space.
    // For the detailed explanation of the following see: Absil, Mahoney, Trumpf, "An extrinsic look
    // at the Riemannian Hessian".

    using namespace Dune::Indices;

    localHessian[_0][_0].setSize(nDofs0,nDofs0);

    for (size_t col=0; col<nDofs0; col++)
    {
      for (size_t subCol=0; subCol<blocksize0; subCol++)
      {
        typename TargetSpace0::EmbeddedTangentVector z = orthonormalFrame0[col][subCol];

        // P_x \partial^2 f z
        for (size_t row=0; row<nDofs0; row++)
        {
          typename TargetSpace0::TangentVector semiEmbeddedProduct;
          embeddedHessian00[row][col].mv(z,semiEmbeddedProduct);

          for (int subRow=0; subRow<blocksize0; subRow++)
            localHessian[_0][_0][row][col][subRow][subCol] = semiEmbeddedProduct[subRow];
        }
      }
    }

    localHessian[_0][_1].setSize(nDofs0,nDofs1);

    for (size_t col=0; col<nDofs1; col++)
    {
      for (size_t subCol=0; subCol<blocksize1; subCol++)
      {
        typename TargetSpace1::EmbeddedTangentVector z = orthonormalFrame1[col][subCol];

        // P_x \partial^2 f z
        for (size_t row=0; row<nDofs0; row++)
        {
          typename TargetSpace0::TangentVector semiEmbeddedProduct;
          embeddedHessian01[row][col].mv(z,semiEmbeddedProduct);

          for (int subRow=0; subRow<blocksize0; subRow++)
            localHessian[_0][_1][row][col][subRow][subCol] = semiEmbeddedProduct[subRow];
        }
      }
    }

    localHessian[_1][_0].setSize(nDofs1,nDofs0);

    for (size_t col=0; col<nDofs0; col++)
    {
      for (size_t subCol=0; subCol<blocksize0; subCol++)
      {
        typename TargetSpace0::EmbeddedTangentVector z = orthonormalFrame0[col][subCol];

        // P_x \partial^2 f z
        for (size_t row=0; row<nDofs1; row++) {
          typename TargetSpace1::TangentVector semiEmbeddedProduct;
          embeddedHessian10[row][col].mv(z,semiEmbeddedProduct);

          for (int subRow=0; subRow<blocksize1; subRow++)
            localHessian[_1][_0][row][col][subRow][subCol] = semiEmbeddedProduct[subRow];
        }
      }
    }

    localHessian[_1][_1].setSize(nDofs1,nDofs1);

    for (size_t col=0; col<nDofs1; col++)
    {
      for (size_t subCol=0; subCol<blocksize1; subCol++)
      {
        typename TargetSpace1::EmbeddedTangentVector z = orthonormalFrame1[col][subCol];

        // P_x \partial^2 f z
        for (size_t row=0; row<nDofs1; row++)
        {
          typename TargetSpace1::TangentVector semiEmbeddedProduct;
          embeddedHessian11[row][col].mv(z,semiEmbeddedProduct);

          for (int subRow=0; subRow<blocksize1; subRow++)
            localHessian[_1][_1][row][col][subRow][subCol] = semiEmbeddedProduct[subRow];
        }
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////
    //  Further correction due to non-planar configuration space
    //  + \mathfrak{A}_x(z,P^\orth_x \partial f)
    //////////////////////////////////////////////////////////////////////////////////////

    // Project embedded gradient onto normal space
    std::vector<typename TargetSpace0::EmbeddedTangentVector> projectedGradient0(nDofs0);
    for (size_t i=0; i<nDofs0; i++)
      projectedGradient0[i] = localConfiguration[_0][i].projectOntoNormalSpace(localEmbeddedGradient0[i]);

    std::vector<typename TargetSpace1::EmbeddedTangentVector> projectedGradient1(nDofs1);
    for (size_t i=0; i<nDofs1; i++)
      projectedGradient1[i] = localConfiguration[_1][i].projectOntoNormalSpace(localEmbeddedGradient1[i]);

    // The Weingarten map has only diagonal entries
    for (size_t row=0; row<nDofs0; row++)
    {
      for (size_t subRow=0; subRow<blocksize0; subRow++)
      {
        typename TargetSpace0::EmbeddedTangentVector z = orthonormalFrame0[row][subRow];
        typename TargetSpace0::EmbeddedTangentVector tmp1 = localConfiguration[_0][row].weingarten(z,projectedGradient0[row]);

        typename TargetSpace0::TangentVector tmp2;
        orthonormalFrame0[row].mv(tmp1,tmp2);

        localHessian[_0][_0][row][row][subRow] += tmp2;
      }
    }

    for (size_t row=0; row<nDofs1; row++)
    {
      for (size_t subRow=0; subRow<blocksize1; subRow++)
      {
        typename TargetSpace1::EmbeddedTangentVector z = orthonormalFrame1[row][subRow];
        typename TargetSpace1::EmbeddedTangentVector tmp1 = localConfiguration[_1][row].weingarten(z,projectedGradient1[row]);

        typename TargetSpace1::TangentVector tmp2;
        orthonormalFrame1[row].mv(tmp1,tmp2);

        localHessian[_1][_1][row][row][subRow] += tmp2;
      }
    }
  }
}

#endif
