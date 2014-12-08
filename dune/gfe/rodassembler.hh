#ifndef DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH
#define DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include <dune/fufem/boundarypatch.hh>

#include "rigidbodymotion.hh"
#include "rodlocalstiffness.hh"
#include "geodesicfeassembler.hh"

/** \brief The FEM operator for an extensible, shearable rod in 3d
 */
template <class GridView, int spaceDim>
class RodAssembler
{
    static_assert(spaceDim==2 || spaceDim==3,
                       "You can only instantiate the class RodAssembler for 2d and 3d spaces");
};

/** \brief The FEM operator for an extensible, shearable rod in 3d
 */
template <class GridView>
class RodAssembler<GridView,3> : public GeodesicFEAssembler<P1NodalBasis<GridView>, RigidBodyMotion<double,3> >
{

    //typedef typename GridType::template Codim<0>::Entity EntityType;
    //typedef typename GridType::template Codim<0>::EntityPointer EntityPointer;
    typedef typename GridView::template Codim<0>::Iterator ElementIterator;

        //! Dimension of the grid.  This needs to be one!
        enum { gridDim = GridView::dimension };

        enum { elementOrder = 1};

        //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
        enum { blocksize = 6 };

        //!
        typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;

public:
        //! ???
    RodAssembler(const GridView &gridView,
                 RodLocalStiffness<GridView,double>* localStiffness)
        : GeodesicFEAssembler<P1NodalBasis<GridView>, RigidBodyMotion<double,3> >(gridView,localStiffness)
        {
            std::vector<RigidBodyMotion<double,3> > referenceConfiguration(gridView.size(gridDim));

            typename GridView::template Codim<gridDim>::Iterator it    = gridView.template begin<gridDim>();
            typename GridView::template Codim<gridDim>::Iterator endIt = gridView.template end<gridDim>();

            for (; it != endIt; ++it) {

                int idx = gridView.indexSet().index(*it);

                referenceConfiguration[idx].r[0] = 0;
                referenceConfiguration[idx].r[1] = 0;
                referenceConfiguration[idx].r[2] = it->geometry().corner(0)[0];
                referenceConfiguration[idx].q = Rotation<double,3>::identity();
            }

            dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->setReferenceConfiguration(referenceConfiguration);
        }

        std::vector<RigidBodyMotion<double,3> > getRefConfig()
        {   return  dynamic_cast<RodLocalStiffness<GridView, double>* >(this->localStiffness_)->referenceConfiguration_;
        }

        void assembleGradient(const std::vector<RigidBodyMotion<double,3> >& sol,
                              Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const;

        void getStrain(const std::vector<RigidBodyMotion<double,3> >& sol,
                       Dune::BlockVector<Dune::FieldVector<double, blocksize> >& strain) const;

        void getStress(const std::vector<RigidBodyMotion<double,3> >& sol,
                       Dune::BlockVector<Dune::FieldVector<double, blocksize> >& stress) const;

        /** \brief Return resultant force across boundary in canonical coordinates

        \note Linear run-time in the size of the grid */
        template <class PatchGridView>
        Dune::FieldVector<double,6> getResultantForce(const BoundaryPatch<PatchGridView>& boundary,
                                                      const std::vector<RigidBodyMotion<double,3> >& sol) const;

    }; // end class


/** \brief The FEM operator for a 2D extensible, shearable rod
 */
template <class GridView>
class RodAssembler<GridView,2> : public GeodesicFEAssembler<P1NodalBasis<GridView>, RigidBodyMotion<double,2> >
{

    typedef typename GridView::template Codim<0>::Entity EntityType;
    typedef typename GridView::template Codim<0>::Iterator ElementIterator;

    //! Dimension of the grid.  This needs to be one!
    enum { gridDim = GridView::dimension };

    enum { elementOrder = 1};

    //! Each block is x, y, theta
    enum { blocksize = 3 };

    //!
    typedef Dune::FieldMatrix<double, blocksize, blocksize> MatrixBlock;

    /** \brief Material constants */
    double B;
    double A1;
    double A3;

public:

    //! ???
    RodAssembler(const GridView &gridView)
        : GeodesicFEAssembler<P1NodalBasis<GridView>, RigidBodyMotion<double,2> >(gridView,NULL)
    {
        B = 1;
        A1 = 1;
        A3 = 1;
    }

    ~RodAssembler() {}

    void setParameters(double b, double a1, double a3) {
        B  = b;
        A1 = a1;
        A3 = a3;
    }

    /** \brief Assemble the tangent stiffness matrix and the right hand side
     */
    void assembleMatrix(const std::vector<RigidBodyMotion<double,2> >& sol,
                        Dune::BCRSMatrix<MatrixBlock>& matrix);

    void assembleGradient(const std::vector<RigidBodyMotion<double,2> >& sol,
                          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const;

    /** \brief Compute the energy of a deformation state */
    double computeEnergy(const std::vector<RigidBodyMotion<double,2> >& sol) const;

protected:

    /** \brief Compute the element tangent stiffness matrix  */
    void getLocalMatrix( EntityType &entity,
                         const std::vector<RigidBodyMotion<double,2> >& localSolution,
                         Dune::Matrix<MatrixBlock>& mat) const;

}; // end class

#include "rodassembler.cc"

#endif

