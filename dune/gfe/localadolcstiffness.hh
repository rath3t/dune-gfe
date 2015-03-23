#ifndef DUNE_GFE_LOCAL_ADOL_C_STIFFNESS_HH
#define DUNE_GFE_LOCAL_ADOL_C_STIFFNESS_HH

#include <adolc/adouble.h>            // use of active doubles
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
// gradient(.) and hessian(.)
#include <adolc/interfaces.h>    // use of "Easy to Use" drivers
#include <adolc/taping.h>             // use of taping

#include <dune/fufem/utilities/adolcnamespaceinjections.hh>

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>

#include <dune/gfe/localfestiffness.hh>

//#define ADOLC_VECTOR_MODE

/** \brief Assembles energy gradient and Hessian with ADOL-C (automatic differentiation)
 */
template<class GridView, class LocalFiniteElement, class VectorType>
class LocalADOLCStiffness
    : public LocalFEStiffness<GridView,LocalFiniteElement,VectorType>
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename VectorType::value_type::field_type RT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    enum {gridDim=GridView::dimension};

    // Hack
    typedef std::vector<Dune::FieldVector<adouble,gridDim> > AVectorType;

public:

    //! Dimension of a tangent space
    enum { blocksize = VectorType::value_type::dimension };

    LocalADOLCStiffness(const LocalFEStiffness<GridView, LocalFiniteElement, AVectorType>* energy)
    : localEnergy_(energy)
    {}

    /** \brief Compute the energy at the current configuration */
    virtual RT energy (const Entity& e,
               const LocalFiniteElement& localFiniteElement,
               const VectorType& localConfiguration,
               const std::vector<Dune::FieldVector<double,3> >& localPointLoads) const;

    /** \brief Assemble the local stiffness matrix at the current position

    This uses the automatic differentiation toolbox ADOL_C.
    */
    virtual void assembleGradientAndHessian(const Entity& e,
                         const LocalFiniteElement& localFiniteElement,
                         const VectorType& localConfiguration,
                         const std::vector<Dune::FieldVector<double,3> >& localPointLoads,
                         VectorType& localGradient);

    const LocalFEStiffness<GridView, LocalFiniteElement, AVectorType>* localEnergy_;

};


template <class GridView, class LocalFiniteElement, class VectorType>
typename LocalADOLCStiffness<GridView, LocalFiniteElement, VectorType>::RT
LocalADOLCStiffness<GridView, LocalFiniteElement, VectorType>::
energy(const Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const VectorType& localSolution,
       const std::vector<Dune::FieldVector<double,3> >& localPointLoads) const
{
    double pureEnergy;

    std::vector<Dune::FieldVector<adouble,blocksize> > localASolution(localSolution.size());

    trace_on(1);

    adouble energy = 0;

    for (size_t i=0; i<localSolution.size(); i++)
      for (size_t j=0; j<localSolution[i].size(); j++)
        localASolution[i][j] <<= localSolution[i][j];

    energy = localEnergy_->energy(element,localFiniteElement,localASolution,localPointLoads);

    energy >>= pureEnergy;

    trace_off();

    return pureEnergy;
}



// ///////////////////////////////////////////////////////////
//   Compute gradient and Hessian together
//   To compute the Hessian we need to compute the gradient anyway, so we may
//   as well return it.  This saves assembly time.
// ///////////////////////////////////////////////////////////
template <class GridType, class LocalFiniteElement, class VectorType>
void LocalADOLCStiffness<GridType, LocalFiniteElement, VectorType>::
assembleGradientAndHessian(const Entity& element,
                const LocalFiniteElement& localFiniteElement,
                const VectorType& localSolution,
                const std::vector<Dune::FieldVector<double,3> >& localPointLoads,
                VectorType& localGradient)
{
    // Tape energy computation.  We may not have to do this every time, but it's comparatively cheap.
    energy(element, localFiniteElement, localSolution, localPointLoads);

    /////////////////////////////////////////////////////////////////
    // Compute the energy gradient
    /////////////////////////////////////////////////////////////////

    // Compute the actual gradient
    size_t nDofs = localSolution.size();
    size_t nDoubles = nDofs*blocksize;
    std::vector<double> xp(nDoubles);
    int idx=0;
    for (size_t i=0; i<nDofs; i++)
        for (size_t j=0; j<blocksize; j++)
            xp[idx++] = localSolution[i][j];

  // Compute gradient
    std::vector<double> g(nDoubles);
    gradient(1,nDoubles,xp.data(),g.data());                  // gradient evaluation

    // Copy into Dune type
    localGradient.resize(localSolution.size());

    idx=0;
    for (size_t i=0; i<nDofs; i++)
        for (size_t j=0; j<blocksize; j++)
            localGradient[i][j] = g[idx++];

    /////////////////////////////////////////////////////////////////
    // Compute Hessian
    /////////////////////////////////////////////////////////////////

    this->A_.setSize(nDofs,nDofs);

#ifndef ADOLC_VECTOR_MODE
    std::vector<double> v(nDoubles);
    std::vector<double> w(nDoubles);

    std::fill(v.begin(), v.end(), 0.0);

    for (size_t i=0; i<nDofs; i++)
      for (size_t ii=0; ii<blocksize; ii++)
      {
        // Evaluate Hessian in the direction of each vector of the orthonormal frame
        for (size_t k=0; k<blocksize; k++)
          v[i*blocksize + k] = (k==ii);

        int rc= 3;
        MINDEC(rc, hess_vec(1, nDoubles, xp.data(), v.data(), w.data()));
        if (rc < 0)
          DUNE_THROW(Dune::Exception, "ADOL-C has returned with error code " << rc << "!");

        for (size_t j=0; j<nDoubles; j++)
          this->A_[i][j/blocksize][ii][j%blocksize] = w[j];

        // Make v the null vector again
        std::fill(&v[i*blocksize], &v[(i+1)*blocksize], 0.0);
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
      for (size_t j=0; j<nDirections; j++)
        this->A_[j/blocksize][i/embeddedBlocksize][j%blocksize][i%embeddedBlocksize] = rawHessian[i][j];

    for(size_t i=0; i<nDoubles; i++) {
        free(rawHessian[i]);
        free(tangent[i]);
    }
#endif

//     std::cout << "ADOL-C stiffness:\n";
//     printmatrix(std::cout, this->A_, "foo", "--");
}

#endif

