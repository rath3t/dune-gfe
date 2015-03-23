#ifndef DUNE_GFE_LOCAL_GEODESIC_FE_ADOL_C_STIFFNESS_HH
#define DUNE_GFE_LOCAL_GEODESIC_FE_ADOL_C_STIFFNESS_HH

#include <adolc/adouble.h>            // use of active doubles
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
// gradient(.) and hessian(.)
#include <adolc/interfaces.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>             // use of taping

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>

#include <dune/gfe/localgeodesicfestiffness.hh>

#define ADOLC_VECTOR_MODE

/** \brief Assembles energy gradient and Hessian with ADOL-C (automatic differentiation)
 */
template<class Basis, class TargetSpace>
class LocalGeodesicFEADOLCStiffness
    : public LocalGeodesicFEStiffness<Basis,TargetSpace>
{
    // grid types
    typedef typename Basis::GridView GridView;
    typedef typename Basis::LocalView::Tree::FiniteElement LocalFiniteElement;
    typedef typename GridView::ctype DT;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    typedef typename TargetSpace::template rebind<adouble>::other ATargetSpace;

    // some other sizes
    enum {gridDim=GridView::dimension};

public:

    //! Dimension of a tangent space
    enum { blocksize = TargetSpace::TangentVector::dimension };

    //! Dimension of the embedding space
    enum { embeddedBlocksize = TargetSpace::EmbeddedTangentVector::dimension };

    LocalGeodesicFEADOLCStiffness(const LocalGeodesicFEStiffness<Basis, ATargetSpace>* energy)
    : localEnergy_(energy)
    {}

    /** \brief Compute the energy at the current configuration */
    virtual RT energy (const Entity& e,
               const LocalFiniteElement& localFiniteElement,
               const std::vector<TargetSpace>& localSolution) const;

    /** \brief Assemble the element gradient of the energy functional

    This uses the automatic differentiation toolbox ADOL_C.
    */
    virtual void assembleGradient(const Entity& element,
                          const LocalFiniteElement& localFiniteElement,
                          const std::vector<TargetSpace>& solution,
                          std::vector<typename TargetSpace::TangentVector>& gradient) const;

    /** \brief Assemble the local stiffness matrix at the current position

    This uses the automatic differentiation toolbox ADOL_C.
    */
    virtual void assembleGradientAndHessian(const Entity& e,
                         const LocalFiniteElement& localFiniteElement,
                         const std::vector<TargetSpace>& localSolution,
                         std::vector<typename TargetSpace::TangentVector>& localGradient);

    const LocalGeodesicFEStiffness<Basis, ATargetSpace>* localEnergy_;

};


template <class Basis, class TargetSpace>
typename LocalGeodesicFEADOLCStiffness<Basis, TargetSpace>::RT
LocalGeodesicFEADOLCStiffness<Basis, TargetSpace>::
energy(const Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<TargetSpace>& localSolution) const
{
    double pureEnergy;

    std::vector<ATargetSpace> localASolution(localSolution.size());

    trace_on(1);

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
      localASolution[i] = aRaw[i];  // may contain a projection onto M -- needs to be done in adouble
    }

    energy = localEnergy_->energy(element,localFiniteElement,localASolution);

    energy >>= pureEnergy;

    trace_off();
#if 0
    size_t tape_stats[STAT_SIZE];
    tapestats(1,tape_stats);             // reading of tape statistics
    cout<<"maxlive "<<tape_stats[NUM_MAX_LIVES]<<"\n";
    cout<<"tay_stack_size "<<tape_stats[TAY_STACK_SIZE]<<"\n";
    cout<<"total number of operations "<<tape_stats[NUM_OPERATIONS]<<"\n";
    // ..... print other tape stats
#endif
    return pureEnergy;
}


template <class Basis, class TargetSpace>
void LocalGeodesicFEADOLCStiffness<Basis, TargetSpace>::
assembleGradient(const Entity& element,
                 const LocalFiniteElement& localFiniteElement,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<typename TargetSpace::TangentVector>& localGradient) const
{
    // Tape energy computation.  We may not have to do this every time, but it's comparatively cheap.
    energy(element, localFiniteElement, localSolution);

    // Compute the actual gradient
    size_t nDofs = localSolution.size();
    size_t nDoubles = nDofs*embeddedBlocksize;
    std::vector<double> xp(nDoubles);
    int idx=0;
    for (size_t i=0; i<nDofs; i++)
        for (size_t j=0; j<embeddedBlocksize; j++)
            xp[idx++] = localSolution[i].globalCoordinates()[j];

  // Compute gradient
    std::vector<double> g(nDoubles);
    gradient(1,nDoubles,xp.data(),g.data());                  // gradient evaluation

    // Copy into Dune type
    std::vector<typename TargetSpace::EmbeddedTangentVector> localEmbeddedGradient(localSolution.size());

    idx=0;
    for (size_t i=0; i<nDofs; i++)
        for (size_t j=0; j<embeddedBlocksize; j++)
            localEmbeddedGradient[i][j] = g[idx++];

//     std::cout << "localEmbeddedGradient:\n";
//     for (size_t i=0; i<nDofs; i++)
//       std::cout << localEmbeddedGradient[i] << std::endl;

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
assembleGradientAndHessian(const Entity& element,
                const LocalFiniteElement& localFiniteElement,
                const std::vector<TargetSpace>& localSolution,
                std::vector<typename TargetSpace::TangentVector>& localGradient)
{
    // Tape energy computation.  We may not have to do this every time, but it's comparatively cheap.
    energy(element, localFiniteElement, localSolution);

    /////////////////////////////////////////////////////////////////
    // Compute the gradient.  It is needed to transform the Hessian
    // into the correct coordinates.
    /////////////////////////////////////////////////////////////////

    // Compute the actual gradient
    size_t nDofs = localSolution.size();
    size_t nDoubles = nDofs*embeddedBlocksize;
    std::vector<double> xp(nDoubles);
    int idx=0;
    for (size_t i=0; i<nDofs; i++)
        for (size_t j=0; j<embeddedBlocksize; j++)
            xp[idx++] = localSolution[i].globalCoordinates()[j];

  // Compute gradient
    std::vector<double> g(nDoubles);
    gradient(1,nDoubles,xp.data(),g.data());                  // gradient evaluation

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

#ifndef ADOLC_VECTOR_MODE
    std::vector<double> v(nDoubles);
    std::vector<double> w(nDoubles);

    std::fill(v.begin(), v.end(), 0.0);

    for (int i=0; i<nDofs; i++)
      for (int ii=0; ii<blocksize; ii++)
      {
        // Evaluate Hessian in the direction of each vector of the orthonormal frame
        for (size_t k=0; k<embeddedBlocksize; k++)
          v[i*embeddedBlocksize + k] = orthonormalFrame[i][ii][k];

        int rc= 3;
        MINDEC(rc, hess_vec(1, nDoubles, xp.data(), v.data(), w.data()));
        if (rc < 0)
          DUNE_THROW(Dune::Exception, "ADOL-C has returned with error code " << rc << "!");

        for (int j=0; j<nDoubles; j++)
          embeddedHessian[i][j/embeddedBlocksize][ii][j%embeddedBlocksize] = w[j];

        // Make v the null vector again
        std::fill(&v[i*embeddedBlocksize], &v[(i+1)*embeddedBlocksize], 0.0);
      }
#else
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

    hess_mat(1,nDoubles,nDirections,xp.data(),tangent,rawHessian);

    // Copy Hessian into Dune data type
    for(size_t i=0; i<nDoubles; i++)
      for (int j=0; j<nDirections; j++)
        embeddedHessian[j/blocksize][i/embeddedBlocksize][j%blocksize][i%embeddedBlocksize] = rawHessian[i][j];

    for(size_t i=0; i<nDoubles; i++) {
        free(rawHessian[i]);
        free(tangent[i]);
    }
#endif

    // From this, compute the Hessian with respect to the manifold (which we assume here is embedded
    // isometrically in a Euclidean space.
    // For the detailed explanation of the following see: Absil, Mahoney, Trumpf, "An extrinsic look
    // at the Riemannian Hessian".

    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
    typedef typename TargetSpace::TangentVector         TangentVector;

    this->A_.setSize(nDofs,nDofs);

    for (size_t col=0; col<nDofs; col++) {

        for (size_t subCol=0; subCol<blocksize; subCol++) {

            EmbeddedTangentVector z = orthonormalFrame[col][subCol];

            // P_x \partial^2 f z
            for (size_t row=0; row<nDofs; row++) {
                TangentVector semiEmbeddedProduct;
                embeddedHessian[row][col].mv(z,semiEmbeddedProduct);

                for (int subRow=0; subRow<blocksize; subRow++)
                    this->A_[row][col][subRow][subCol] = semiEmbeddedProduct[subRow];
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

        this->A_[row][row][subRow] += tmp2;
      }

    }

//     std::cout << "ADOL-C stiffness:\n";
//     printmatrix(std::cout, this->A_, "foo", "--");
}

#endif

