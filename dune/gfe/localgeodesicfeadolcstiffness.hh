#ifndef DUNE_GFE_LOCAL_GEODESIC_FE_ADOL_C_STIFFNESS_HH
#define DUNE_GFE_LOCAL_GEODESIC_FE_ADOL_C_STIFFNESS_HH

#include <adolc/adouble.h>            // use of active doubles
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
// gradient(.) and hessian(.)
#include <adolc/taping.h>             // use of taping

#undef overwrite  // stupid: ADOL-C sets this to 1, so the name cannot be used

#include <dune/gfe/adolcnamespaceinjections.hh>

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>

#include <dune/gfe/localgeodesicfestiffness.hh>

/** \brief Assembles energy gradient and Hessian with ADOL-C (automatic differentiation)
 */
template<class GridView, class LocalFiniteElement, class TargetSpace>
class LocalGeodesicFEADOLCStiffness
    : public LocalGeodesicFEStiffness<GridView,LocalFiniteElement,TargetSpace>
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
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

    LocalGeodesicFEADOLCStiffness(const LocalGeodesicFEStiffness<GridView, LocalFiniteElement, ATargetSpace>* energy)
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
    virtual void assembleHessian(const Entity& e,
                         const LocalFiniteElement& localFiniteElement,
                         const std::vector<TargetSpace>& localSolution);

    const LocalGeodesicFEStiffness<GridView, LocalFiniteElement, ATargetSpace>* localEnergy_;

};


template <class GridView, class LocalFiniteElement, class TargetSpace>
typename LocalGeodesicFEADOLCStiffness<GridView, LocalFiniteElement, TargetSpace>::RT
LocalGeodesicFEADOLCStiffness<GridView, LocalFiniteElement, TargetSpace>::
energy(const Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<TargetSpace>& localSolution) const
{
    double pureEnergy;

    std::vector<ATargetSpace> localASolution(localSolution.size());

    trace_on(1);

    adouble energy = 0;

    for (size_t i=0; i<localSolution.size(); i++)
      localASolution[i] <<=localSolution[i];

    energy = localEnergy_->energy(element,localFiniteElement,localASolution);

    energy >>= pureEnergy;

    trace_off();
#if 0
    size_t tape_stats[STAT_SIZE];
    tapestats(1,tape_stats);             // reading of tape statistics
    cout<<"maxlive "<<tape_stats[NUM_MAX_LIVES]<<"\n";
    // ..... print other tape stats
#endif
    return pureEnergy;
}


template <class GridView, class LocalFiniteElement, class TargetSpace>
void LocalGeodesicFEADOLCStiffness<GridView, LocalFiniteElement, TargetSpace>::
assembleGradient(const Entity& element,
                 const LocalFiniteElement& localFiniteElement,
                 const std::vector<TargetSpace>& localSolution,
                 std::vector<typename TargetSpace::TangentVector>& localGradient) const
{
    double pureEnergy;

    std::vector<ATargetSpace> localASolution(localSolution.size());

    trace_on(1);

    adouble energy = 0;

    for (size_t i=0; i<localSolution.size(); i++)
      localASolution[i] <<=localSolution[i];

    energy = localEnergy_->energy(element,localFiniteElement,localASolution);

    energy >>= pureEnergy;

    trace_off(0);
    //std::cout << "ADOL-C energy: " << pureEnergy << std::endl;
#if 0
    size_t tape_stats[STAT_SIZE];
    tapestats(1,tape_stats);             // reading of tape statistics
    cout<<"maxlive "<<tape_stats[NUM_MAX_LIVES]<<"\n";
    // ..... print other tape stats
#endif
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
//   Compute gradient by finite-difference approximation
// ///////////////////////////////////////////////////////////
template <class GridType, class LocalFiniteElement, class TargetSpace>
void LocalGeodesicFEADOLCStiffness<GridType, LocalFiniteElement, TargetSpace>::
assembleHessian(const Entity& element,
                const LocalFiniteElement& localFiniteElement,
                const std::vector<TargetSpace>& localSolution)
{
    double pureEnergy;

    std::vector<ATargetSpace> localASolution(localSolution.size());

    trace_on(1,1);

    adouble energy = 0;

    for (size_t i=0; i<localSolution.size(); i++)
      localASolution[i] <<=localSolution[i];

    energy = localEnergy_->energy(element,localFiniteElement,localASolution);

    energy >>= pureEnergy;

    trace_off(0);

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

    /////////////////////////////////////////////////////////////////
    // Compute Hessian
    /////////////////////////////////////////////////////////////////
    double** rawHessian = (double**) malloc(nDoubles*sizeof(double*));
    for(size_t i=0; i<nDoubles; i++)
        rawHessian[i] = (double*)malloc((i+1)*sizeof(double));
    hessian(1,nDoubles,xp.data(),rawHessian);

    // Copy Hessian into Dune data type
    Dune::Matrix<Dune::FieldMatrix<double,embeddedBlocksize, embeddedBlocksize> > embeddedHessian(nDofs,nDofs);
    for(size_t i=0; i<nDoubles; i++) {
      for (size_t j=0; j<nDoubles; j++) {
        double value = (j<=i) ? rawHessian[i][j] : rawHessian[j][i];
        embeddedHessian[i/embeddedBlocksize][j/embeddedBlocksize][i%embeddedBlocksize][j%embeddedBlocksize] = value;
      }
    }

    // From this, compute the Hessian with respect to the manifold (which we assume here is embedded
    // isometrically in a Euclidean space.
    // For the detailed explanation of the following see: Absil, Mahoney, Trumpf, "An extrinsic look
    // at the Riemannian Hessian".

    typedef typename TargetSpace::EmbeddedTangentVector EmbeddedTangentVector;
    typedef typename TargetSpace::TangentVector         TangentVector;

    this->A_.setSize(nDofs,nDofs);

    std::vector<Dune::FieldMatrix<RT,blocksize,embeddedBlocksize> > orthonormalFrame(nDofs);

    for (size_t i=0; i<nDofs; i++)
        orthonormalFrame[i] = localSolution[i].orthonormalFrame();

    for (size_t col=0; col<nDofs; col++) {

        for (size_t subCol=0; subCol<blocksize; subCol++) {

            EmbeddedTangentVector z = orthonormalFrame[col][subCol];

            // P_x \partial^2 f z
            std::vector<EmbeddedTangentVector> semiEmbeddedProduct(nDofs);

            for (size_t row=0; row<nDofs; row++) {
                embeddedHessian[row][col].mv(z,semiEmbeddedProduct[row]);
                //tmp1 = localSolution[row].projectOntoTangentSpace(tmp1);
                TangentVector tmp2;
                orthonormalFrame[row].mv(semiEmbeddedProduct[row],tmp2);

                for (int subRow=0; subRow<blocksize; subRow++)
                    this->A_[row][col][subRow][subCol] = tmp2[subRow];
            }

            // + A_x (z, P_x^\orth \partial f)
            for (size_t row=0; row<nDofs; row++) {

                // Get the Euclidean gradient projected onto the normal space
                EmbeddedTangentVector porthGrad = localSolution[row].projectOntoNormalSpace(localEmbeddedGradient[row]);

                EmbeddedTangentVector tmp3 = localSolution[row].weingarten(z, porthGrad);

                TangentVector tmp4;
                orthonormalFrame[row].mv(tmp3,tmp4);

                for (int subRow=0; subRow<blocksize; subRow++)
                    this->A_[row][col][subRow][subCol] += tmp4[subRow];
            }

        }

    }

//     std::cout << "ADOL-C stiffness:\n";
//     printmatrix(std::cout, this->A_, "foo", "--");
}

#endif

