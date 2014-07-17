#ifndef DUNE_GFE_GRAMSCHMIDTSOLVER_HH
#define DUNE_GFE_GRAMSCHMIDTSOLVER_HH

#include <dune/common/fvector.hh>

#include <dune/gfe/symmetricmatrix.hh>

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
  static void normalize(const Dune::SymmetricMatrix<field_type,embeddedDim>& matrix,
                        Dune::FieldVector<field_type,embeddedDim>& v)
  {
    v /= std::sqrt(matrix.energyScalarProduct(v,v));
  }


  /** \brief Project vj on vi, and subtract the result from vj
   *
   * \param matrix The matrix the defines the scalar product
   */
  static void project(const Dune::SymmetricMatrix<field_type,embeddedDim>& matrix,
                      const Dune::FieldVector<field_type,embeddedDim>& vi,
                      Dune::FieldVector<field_type,embeddedDim>& vj)
  {

    field_type energyScalarProduct = matrix.energyScalarProduct(vi,vj);

    for (size_t i=0; i<vj.size(); i++)
      vj[i] -= energyScalarProduct * vi[i];

  }

public:

  /** \brief Constructor computing the A-orthogonal basis
   *
   * This constructor uses Gram-Schmidt orthogonalization to compute an A-orthogonal basis.
   * All (non-static) calls to 'solve' will use that basis.  Since computing the basis
   * is the expensive part, the calls to 'solve' will be comparatively cheap.
   *
   * \param basis Any basis of the space, used as the input for the Gram-Schmidt orthogonalization process
   */
  GramSchmidtSolver(const Dune::SymmetricMatrix<field_type,embeddedDim>& matrix,
                    const Dune::FieldMatrix<field_type,dim,embeddedDim>& basis)
  : orthonormalBasis_(basis)
  {
    // Use the Gram-Schmidt algorithm to compute a basis that is orthonormal
    // with respect to the given matrix.
    for (int i=0; i<dim; i++) {

      normalize(matrix, orthonormalBasis_[i]);

      for (int j=0; j<i; j++)
        project(matrix, orthonormalBasis_[i], orthonormalBasis_[j]);

    }

  }

  void solve(Dune::FieldVector<field_type,embeddedDim>& x,
             const Dune::FieldVector<field_type,embeddedDim>& rhs) const
  {
    // Solve the system in the orthonormal basis
    Dune::FieldVector<field_type,dim> orthoCoefficient;
    for (int i=0; i<dim; i++)
      orthoCoefficient[i] = rhs*orthonormalBasis_[i];

    // Solution in canonical basis
    x = 0;
    for (int i=0; i<dim; i++)
      x.axpy(orthoCoefficient[i], orthonormalBasis_[i]);
  }

  /** Solve linear system by constructing an energy-orthonormal basis

   * \param basis Any basis of the space, used as the input for the Gram-Schmidt orthogonalization process
   */
  static void solve(const Dune::SymmetricMatrix<field_type,embeddedDim>& matrix,
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

private:

  Dune::FieldMatrix<field_type,dim,embeddedDim> orthonormalBasis_;

};

#endif