#ifndef DUNE_GFE_MIXEDLOCALGFEADOLCSTIFFNESS_HH
#define DUNE_GFE_MIXEDLOCALGFEADOLCSTIFFNESS_HH

#include <adolc/adouble.h>            // use of active doubles
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
// gradient(.) and hessian(.)
#include <adolc/interfaces.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>             // use of taping

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>

#include <dune/gfe/assemblers/mixedlocalgeodesicfestiffness.hh>
#include <dune/gfe/spaces/productmanifold.hh>

/** \brief Assembles energy gradient and Hessian with ADOL-C (automatic differentiation)
 */
template<class Basis, class TargetSpace0, class TargetSpace1>
class MixedLocalGFEADOLCStiffness
  : public MixedLocalGeodesicFEStiffness<Basis,Dune::GFE::ProductManifold<TargetSpace0,TargetSpace1> >
{
  // grid types
  typedef typename Basis::GridView GridView;
  typedef typename GridView::ctype DT;
  typedef typename TargetSpace0::ctype RT;
  typedef typename GridView::template Codim<0>::Entity Entity;

  // The 'active' target spaces, i.e., the number type is replaced by adouble
  typedef typename TargetSpace0::template rebind<adouble>::other ATargetSpace0;
  typedef typename TargetSpace1::template rebind<adouble>::other ATargetSpace1;

  // some other sizes
  constexpr static int gridDim = GridView::dimension;

public:

  //! Dimension of a tangent space
  constexpr static int blocksize0 = TargetSpace0::TangentVector::dimension;
  constexpr static int blocksize1 = TargetSpace1::TangentVector::dimension;

  //! Dimension of the embedding space
  constexpr static int embeddedBlocksize0 = TargetSpace0::EmbeddedTangentVector::dimension;
  constexpr static int embeddedBlocksize1 = TargetSpace1::EmbeddedTangentVector::dimension;

  MixedLocalGFEADOLCStiffness(const MixedLocalGeodesicFEStiffness<Basis, Dune::GFE::ProductManifold<ATargetSpace0,
      ATargetSpace1> >* energy, bool adolcScalarMode = false)
    : localEnergy_(energy),
    adolcScalarMode_(adolcScalarMode)
  {}

  /** \brief Compute the energy at the current configuration */
  virtual RT energy (const typename Basis::LocalView& localView,
                     const std::vector<TargetSpace0>& localConfiguration0,
                     const std::vector<TargetSpace1>& localConfiguration1) const;

  /** \brief Assemble the local stiffness matrix at the current position

     This uses the automatic differentiation toolbox ADOL_C.
   */
  virtual void assembleGradientAndHessian(const typename Basis::LocalView& localView,
                                          const std::vector<TargetSpace0>& localConfiguration0,
                                          const std::vector<TargetSpace1>& localConfiguration1,
                                          std::vector<typename TargetSpace0::TangentVector>& localGradient0,
                                          std::vector<typename TargetSpace1::TangentVector>& localGradient1) override;

  const MixedLocalGeodesicFEStiffness<Basis, Dune::GFE::ProductManifold<ATargetSpace0, ATargetSpace1> >* localEnergy_;
  const bool adolcScalarMode_;
};


template <class Basis, class TargetSpace0, class TargetSpace1>
typename MixedLocalGFEADOLCStiffness<Basis, TargetSpace0, TargetSpace1>::RT
MixedLocalGFEADOLCStiffness<Basis, TargetSpace0, TargetSpace1>::
energy(const typename Basis::LocalView& localView,
       const std::vector<TargetSpace0>& localConfiguration0,
       const std::vector<TargetSpace1>& localConfiguration1) const
{
  int rank = Dune::MPIHelper::getCommunication().rank();
  double pureEnergy;

  std::vector<ATargetSpace0> localAConfiguration0(localConfiguration0.size());
  std::vector<ATargetSpace1> localAConfiguration1(localConfiguration1.size());

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
  std::vector<typename ATargetSpace0::CoordinateType> aRaw0(localConfiguration0.size());
  for (size_t i=0; i<localConfiguration0.size(); i++) {
    typename TargetSpace0::CoordinateType raw = localConfiguration0[i].globalCoordinates();
    for (size_t j=0; j<raw.size(); j++)
      aRaw0[i][j] <<= raw[j];
    localAConfiguration0[i] = aRaw0[i];    // may contain a projection onto M -- needs to be done in adouble
  }

  std::vector<typename ATargetSpace1::CoordinateType> aRaw1(localConfiguration1.size());
  for (size_t i=0; i<localConfiguration1.size(); i++) {
    typename TargetSpace1::CoordinateType raw = localConfiguration1[i].globalCoordinates();
    for (size_t j=0; j<raw.size(); j++)
      aRaw1[i][j] <<= raw[j];
    localAConfiguration1[i] = aRaw1[i];    // may contain a projection onto M -- needs to be done in adouble
  }

  using namespace Dune::TypeTree::Indices;
  try {
    energy = localEnergy_->energy(localView,
                                  localAConfiguration0,
                                  localAConfiguration1);
  } catch (Dune::Exception &e) {
    trace_off();
    throw e;
  }
  energy >>= pureEnergy;

  trace_off();

  return pureEnergy;
}

// ///////////////////////////////////////////////////////////
//   Compute gradient and Hessian together
//   To compute the Hessian we need to compute the gradient anyway, so we may
//   as well return it.  This saves assembly time.
// ///////////////////////////////////////////////////////////
template <class Basis, class TargetSpace0, class TargetSpace1>
void MixedLocalGFEADOLCStiffness<Basis, TargetSpace0, TargetSpace1>::
assembleGradientAndHessian(const typename Basis::LocalView& localView,
                           const std::vector<TargetSpace0>& localConfiguration0,
                           const std::vector<TargetSpace1>& localConfiguration1,
                           std::vector<typename TargetSpace0::TangentVector>& localGradient0,
                           std::vector<typename TargetSpace1::TangentVector>& localGradient1)
{
  int rank = Dune::MPIHelper::getCommunication().rank();
  // Tape energy computation.  We may not have to do this every time, but it's comparatively cheap.
  energy(localView, localConfiguration0, localConfiguration1);

  /////////////////////////////////////////////////////////////////
  // Compute the gradient.  It is needed to transform the Hessian
  // into the correct coordinates.
  /////////////////////////////////////////////////////////////////

  // Compute the actual gradient
  size_t nDofs0 = localConfiguration0.size();
  size_t nDofs1 = localConfiguration1.size();
  size_t nDoubles = nDofs0*embeddedBlocksize0 + nDofs1*embeddedBlocksize1;

  std::vector<double> xp(nDoubles);
  int idx=0;
  for (size_t i=0; i<localConfiguration0.size(); i++)
    for (size_t j=0; j<embeddedBlocksize0; j++)
      xp[idx++] = localConfiguration0[i].globalCoordinates()[j];

  for (size_t i=0; i<localConfiguration1.size(); i++)
    for (size_t j=0; j<embeddedBlocksize1; j++)
      xp[idx++] = localConfiguration1[i].globalCoordinates()[j];

  // Compute gradient
  std::vector<double> g(nDoubles);
  gradient(rank,nDoubles,xp.data(),g.data());                    // gradient evaluation

  // Copy into Dune type
  std::vector<typename TargetSpace0::EmbeddedTangentVector> localEmbeddedGradient0(localConfiguration0.size());
  std::vector<typename TargetSpace1::EmbeddedTangentVector> localEmbeddedGradient1(localConfiguration1.size());

  idx=0;
  for (size_t i=0; i<localConfiguration0.size(); i++) {
    for (size_t j=0; j<embeddedBlocksize0; j++)
      localEmbeddedGradient0[i][j] = g[idx++];

    // Express gradient in local coordinate system
    localConfiguration0[i].orthonormalFrame().mv(localEmbeddedGradient0[i],localGradient0[i]);
  }

  for (size_t i=0; i<localConfiguration1.size(); i++) {
    for (size_t j=0; j<embeddedBlocksize1; j++)
      localEmbeddedGradient1[i][j] = g[idx++];

    // Express gradient in local coordinate system
    localConfiguration1[i].orthonormalFrame().mv(localEmbeddedGradient1[i],localGradient1[i]);
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
    orthonormalFrame0[i] = localConfiguration0[i].orthonormalFrame();

  std::vector<Dune::FieldMatrix<RT,blocksize1,embeddedBlocksize1> > orthonormalFrame1(nDofs1);

  for (size_t i=0; i<nDofs1; i++)
    orthonormalFrame1[i] = localConfiguration1[i].orthonormalFrame();

  Dune::Matrix<Dune::FieldMatrix<double,blocksize0, embeddedBlocksize0> > embeddedHessian00(nDofs0,nDofs0);
  Dune::Matrix<Dune::FieldMatrix<double,blocksize0, embeddedBlocksize1> > embeddedHessian01(nDofs0,nDofs1);
  Dune::Matrix<Dune::FieldMatrix<double,blocksize1, embeddedBlocksize0> > embeddedHessian10(nDofs1,nDofs0);
  Dune::Matrix<Dune::FieldMatrix<double,blocksize1, embeddedBlocksize1> > embeddedHessian11(nDofs1,nDofs1);

  if(adolcScalarMode_) {
    std::vector<double> v(nDoubles);
    std::vector<double> w(nDoubles);

    std::fill(v.begin(), v.end(), 0.0);

    size_t nDoubles0 = nDofs0*embeddedBlocksize0;     // nDoubles = nDoubles0 + nDoubles1
    size_t nDoubles1 = nDofs1*embeddedBlocksize1;

    std::fill(v.begin(), v.end(), 0.0);

    for (size_t i=0; i<nDofs0 + nDofs1; i++) {

      // Evaluate Hessian in the direction of each vector of the orthonormal frame
      if (i < nDofs0) {       //Upper half
        auto i0 = i;
        for (int ii0=0; ii0<blocksize0; ii0++) {
          for (size_t k0=0; k0<embeddedBlocksize0; k0++) {
            v[i0*embeddedBlocksize0 + k0] = orthonormalFrame0[i0][ii0][k0];
          }
          int rc= 3;
          MINDEC(rc, hess_vec(rank, nDoubles, xp.data(), v.data(), w.data()));
          if (rc < 0)
            DUNE_THROW(Dune::Exception, "ADOL-C has returned with error code " << rc << "!");

          for (size_t j0=0; j0<nDoubles0; j0++)           //Upper left
            embeddedHessian00[i0][j0/embeddedBlocksize0][ii0][j0%embeddedBlocksize0] = w[j0];

          for (size_t j1=0; j1<nDoubles1; j1++)           //Upper right
            embeddedHessian01[i0][j1/embeddedBlocksize1][ii0][j1%embeddedBlocksize1] = w[nDoubles0 + j1];

          // Make v the null vector again
          std::fill(&v[i0*embeddedBlocksize0], &v[(i0+1)*embeddedBlocksize0], 0.0);
        }
      } else  {       // i = nDofs0 ... nDofs0 + nDofs1 //lower half
        auto i1 = i - nDofs0;
        for (int ii1=0; ii1<blocksize1; ii1++) {
          for (size_t k1=0; k1<embeddedBlocksize1; k1++) {
            v[nDoubles0 + i1*embeddedBlocksize1 + k1] = orthonormalFrame1[i1][ii1][k1];
          }
          int rc= 3;
          MINDEC(rc, hess_vec(rank, nDoubles, xp.data(), v.data(), w.data()));
          if (rc < 0)
            DUNE_THROW(Dune::Exception, "ADOL-C has returned with error code " << rc << "!");

          for (size_t j0=0; j0<nDoubles0; j0++)           // Upper left
            embeddedHessian10[i1][j0/embeddedBlocksize0][ii1][j0%embeddedBlocksize0] = w[j0];

          for (size_t j1=0; j1<nDoubles1; j1++)           //Upper right
            embeddedHessian11[i1][j1/embeddedBlocksize1][ii1][j1%embeddedBlocksize1] = w[nDoubles0 + j1];

          // Make v the null vector again
          std::fill(&v[nDoubles0 + i1*embeddedBlocksize1], &v[nDoubles0 + (i1+1)*embeddedBlocksize1], 0.0);
        }
      }
    }

  } else {   //ADOL-C vector mode}
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

  this->A00_.setSize(nDofs0,nDofs0);

  for (size_t col=0; col<nDofs0; col++) {

    for (size_t subCol=0; subCol<blocksize0; subCol++) {

      typename TargetSpace0::EmbeddedTangentVector z = orthonormalFrame0[col][subCol];

      // P_x \partial^2 f z
      for (size_t row=0; row<nDofs0; row++) {
        typename TargetSpace0::TangentVector semiEmbeddedProduct;
        embeddedHessian00[row][col].mv(z,semiEmbeddedProduct);

        for (int subRow=0; subRow<blocksize0; subRow++)
          this->A00_[row][col][subRow][subCol] = semiEmbeddedProduct[subRow];
      }

    }

  }

  this->A01_.setSize(nDofs0,nDofs1);

  for (size_t col=0; col<nDofs1; col++) {

    for (size_t subCol=0; subCol<blocksize1; subCol++) {

      typename TargetSpace1::EmbeddedTangentVector z = orthonormalFrame1[col][subCol];

      // P_x \partial^2 f z
      for (size_t row=0; row<nDofs0; row++) {
        typename TargetSpace0::TangentVector semiEmbeddedProduct;
        embeddedHessian01[row][col].mv(z,semiEmbeddedProduct);

        for (int subRow=0; subRow<blocksize0; subRow++)
          this->A01_[row][col][subRow][subCol] = semiEmbeddedProduct[subRow];
      }

    }

  }

  this->A10_.setSize(nDofs1,nDofs0);

  for (size_t col=0; col<nDofs0; col++) {

    for (size_t subCol=0; subCol<blocksize0; subCol++) {

      typename TargetSpace0::EmbeddedTangentVector z = orthonormalFrame0[col][subCol];

      // P_x \partial^2 f z
      for (size_t row=0; row<nDofs1; row++) {
        typename TargetSpace1::TangentVector semiEmbeddedProduct;
        embeddedHessian10[row][col].mv(z,semiEmbeddedProduct);

        for (int subRow=0; subRow<blocksize1; subRow++)
          this->A10_[row][col][subRow][subCol] = semiEmbeddedProduct[subRow];
      }

    }

  }

  this->A11_.setSize(nDofs1,nDofs1);

  for (size_t col=0; col<nDofs1; col++) {

    for (size_t subCol=0; subCol<blocksize1; subCol++) {

      typename TargetSpace1::EmbeddedTangentVector z = orthonormalFrame1[col][subCol];

      // P_x \partial^2 f z
      for (size_t row=0; row<nDofs1; row++) {
        typename TargetSpace1::TangentVector semiEmbeddedProduct;
        embeddedHessian11[row][col].mv(z,semiEmbeddedProduct);

        for (int subRow=0; subRow<blocksize1; subRow++)
          this->A11_[row][col][subRow][subCol] = semiEmbeddedProduct[subRow];
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
    projectedGradient0[i] = localConfiguration0[i].projectOntoNormalSpace(localEmbeddedGradient0[i]);

  std::vector<typename TargetSpace1::EmbeddedTangentVector> projectedGradient1(nDofs1);
  for (size_t i=0; i<nDofs1; i++)
    projectedGradient1[i] = localConfiguration1[i].projectOntoNormalSpace(localEmbeddedGradient1[i]);

  // The Weingarten map has only diagonal entries
  for (size_t row=0; row<nDofs0; row++) {

    for (size_t subRow=0; subRow<blocksize0; subRow++) {

      typename TargetSpace0::EmbeddedTangentVector z = orthonormalFrame0[row][subRow];
      typename TargetSpace0::EmbeddedTangentVector tmp1 = localConfiguration0[row].weingarten(z,projectedGradient0[row]);

      typename TargetSpace0::TangentVector tmp2;
      orthonormalFrame0[row].mv(tmp1,tmp2);

      this->A00_[row][row][subRow] += tmp2;
    }

  }

  for (size_t row=0; row<nDofs1; row++) {

    for (size_t subRow=0; subRow<blocksize1; subRow++) {

      typename TargetSpace1::EmbeddedTangentVector z = orthonormalFrame1[row][subRow];
      typename TargetSpace1::EmbeddedTangentVector tmp1 = localConfiguration1[row].weingarten(z,projectedGradient1[row]);

      typename TargetSpace1::TangentVector tmp2;
      orthonormalFrame1[row].mv(tmp1,tmp2);

      this->A11_[row][row][subRow] += tmp2;
    }

  }
}

#endif
