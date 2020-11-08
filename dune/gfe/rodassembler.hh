#ifndef DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH
#define DUNE_EXTENSIBLE_ROD_ASSEMBLER_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrix.hh>

#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/localgeodesicfefdstiffness.hh>
#include "rigidbodymotion.hh"
#include "rodlocalstiffness.hh"
#include "geodesicfeassembler.hh"

/** \brief The FEM operator for an extensible, shearable rod in 3d
 */
template <class Basis, int spaceDim>
class RodAssembler
{
    static_assert(spaceDim==2 || spaceDim==3,
                       "You can only instantiate the class RodAssembler for 2d and 3d spaces");
};

/** \brief The FEM operator for an extensible, shearable rod in 3d
 */
template <class Basis>
class RodAssembler<Basis,3> : public GeodesicFEAssembler<Basis, RigidBodyMotion<double,3> >
{
  typedef typename Basis::GridView GridView;

  //! Dimension of the grid.
  enum { gridDim = GridView::dimension };
  static_assert(gridDim==1, "RodAssembler can only be used with one-dimensional grids!");

        enum { elementOrder = 1};

        //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
        enum { blocksize = 6 };

public:
        //! ???
    RodAssembler(const Basis& basis,
                 LocalGeodesicFEStiffness<Basis, RigidBodyMotion<double,3> >& localStiffness)
    : GeodesicFEAssembler<Basis, RigidBodyMotion<double,3> >(basis,localStiffness)
        {
            std::vector<RigidBodyMotion<double,3> > referenceConfiguration(basis.size());

    for (const auto vertex : Dune::vertices(basis.gridView()))
    {
      auto idx = basis.gridView().indexSet().index(vertex);

                referenceConfiguration[idx].r[0] = 0;
                referenceConfiguration[idx].r[1] = 0;
      referenceConfiguration[idx].r[2] = vertex.geometry().corner(0)[0];
                referenceConfiguration[idx].q = Rotation<double,3>::identity();
            }

    rodEnergy()->setReferenceConfiguration(referenceConfiguration);
        }

    auto rodEnergy()
    {
      // TODO: Does not work for other stiffness implementations
      auto localFDStiffness = std::dynamic_pointer_cast<LocalGeodesicFEFDStiffness<Basis, RigidBodyMotion<double,3> > >(this->localStiffness_);
      return const_cast<RodLocalStiffness<GridView,double>*>(dynamic_cast<const RodLocalStiffness<GridView,double>*>(localFDStiffness->localEnergy_));
    }

        std::vector<RigidBodyMotion<double,3> > getRefConfig()
    {
      return rodEnergy()->referenceConfiguration_;
        }

  virtual void assembleGradient(const std::vector<RigidBodyMotion<double,3> >& sol,
                                Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const override;

    }; // end class


/** \brief The FEM operator for a 2D extensible, shearable rod
 */
template <class Basis>
class RodAssembler<Basis,2> : public GeodesicFEAssembler<Basis, RigidBodyMotion<double,2> >
{

    typedef typename Basis::GridView GridView;
    typedef typename GridView::template Codim<0>::Entity EntityType;

  //! Dimension of the grid.
  enum { gridDim = GridView::dimension };
  static_assert(gridDim==1, "RodAssembler can only be used with one-dimensional grids!");

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
        : GeodesicFEAssembler<Basis, RigidBodyMotion<double,2> >(gridView,nullptr)
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

    virtual void assembleGradient(const std::vector<RigidBodyMotion<double,2> >& sol,
                          Dune::BlockVector<Dune::FieldVector<double, blocksize> >& grad) const override;

    /** \brief Compute the energy of a deformation state */
    virtual double computeEnergy(const std::vector<RigidBodyMotion<double,2> >& sol) const override;

protected:

    /** \brief Compute the element tangent stiffness matrix  */
    void getLocalMatrix( EntityType &entity,
                         const std::vector<RigidBodyMotion<double,2> >& localSolution,
                         Dune::Matrix<MatrixBlock>& mat) const;

}; // end class

#include "rodassembler.cc"

#endif

