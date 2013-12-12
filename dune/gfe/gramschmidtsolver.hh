#ifndef DUNE_GFE_GRAMSCHMIDTSOLVER_HH
#define DUNE_GFE_GRAMSCHMIDTSOLVER_HH

#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>


/** \brief Direct solver for a dense symmetric linear system, using an orthonormal basis
 *
 * This solver computes an A-orthonormal basis, and uses that to compute the solution
 * of a linear system.  The advantage of this is that it works even if the matrix is
 * known to have a non-trivial kernel.  This happens in GFE applications when the Hessian
 * of a functional on a manifold is given in coordinates of the surrounding space.
 * The method if efficient for the small systems that we consider in GFE applications.
 *
 * \tparam field_type Type used to store scalars
 * \tparam embeddedDim Number of rows and columns of the linear system
 * \tparam dim The rank of the matrix
 */
template <class field_type, int dim, int embeddedDim>
class GramSchmidtSolver
{
  /** \brief Normalize a vector to unit length, measured in a matrix norm
   * \param matrix The matrix inducing the matrix norm
   * \param[in,out] v The vector to normalize
   */
  static void normalize(const Dune::FieldMatrix<field_type,embeddedDim,embeddedDim>& matrix,
                        Dune::FieldVector<field_type,embeddedDim>& v)
  {
    field_type energyNormSquared = 0;

    for (int i=0; i<v.size(); i++)
      for (int j=0; j<v.size(); j++)
        energyNormSquared += matrix[i][j]*v[i]*v[j];

    v /= std::sqrt(energyNormSquared);
  }


  /** \brief Project vj on vi, and subtract the result from vj
   *
   * \param matrix The matrix the defines the scalar product
   */
  static void project(const Dune::FieldMatrix<field_type,embeddedDim,embeddedDim>& matrix,
                      const Dune::FieldVector<field_type,embeddedDim>& vi,
                      Dune::FieldVector<field_type,embeddedDim>& vj)
  {

    field_type energyScalarProduct = 0;

    for (int i=0; i<vi.size(); i++)
      for (int j=0; j<vj.size(); j++)
        energyScalarProduct += matrix[i][j]*vi[i]*vj[j];

    for (int i=0; i<vj.size(); i++)
      vj[i] -= energyScalarProduct * vi[i];

  }

public:
  /** Solve linear system by constructing an energy-orthonormal basis */
  static void solve(const Dune::FieldMatrix<field_type,embeddedDim,embeddedDim>& matrix,
                    Dune::FieldVector<field_type,embeddedDim>& x,
                    const Dune::FieldVector<field_type,embeddedDim>& rhs,
                    const Dune::FieldMatrix<field_type,dim,embeddedDim>& basis)
  {
    // Use the Gram-Schmidt algorithm to compute a basis that is orthonormal
    // with respect to the given matrix.
    Dune::FieldMatrix<field_type,dim,embeddedDim> orthonormalBasis(basis);

    for (int i=0; i<dim; i++) {

      normalize(matrix, orthonormalBasis[i]);

      for (int j=0; j<i; j++)
        project(matrix, orthonormalBasis[i], orthonormalBasis[j]);

    }

    // Solve the system in the orthonormal basis
    Dune::FieldVector<field_type,dim> orthoCoefficient;
    for (int i=0; i<dim; i++)
      orthoCoefficient[i] = rhs*orthonormalBasis[i];

    // Solution in canonical basis
    x = 0;
    for (int i=0; i<dim; i++)
      x.axpy(orthoCoefficient[i], orthonormalBasis[i]);

  }

};

#endif