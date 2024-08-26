#ifndef DUNE_GFE_COSSERATSTRAIN_HH
#define DUNE_GFE_COSSERATSTRAIN_HH

#include <dune/common/fmatrix.hh>

namespace Dune {

  namespace GFE {

    /** \brief Strain tensor of a Cosserat material
     *
     * Mathematically, this object is a dimworld x dimworld matrix.
     * However, the last dimworld-dim columns consists of the corresponding
     * columns of the identity matrix.  Knowing this allows more efficient
     * implementations of several frequently occurring operations.
     *
     * \tparam T Type used for real numbers
     * \tparam dimworld Dimension of the world space
     * \tparam dim Dimension of the domain
     */
    template <class T, int dimworld, int dim>
    class CosseratStrain
    {
    public:
      /** \brief Compute strain from the deformation gradient and the microrotation
       *
       * \param R The microrotation
       */
      CosseratStrain(const FieldMatrix<T,dimworld+4,dim>& deformationGradient,
                     const FieldMatrix<T,dimworld,dimworld>& R)
      {
        /* Compute F, the deformation gradient.
           In the continuum case (domain dimension == world dimension) this is
           \f$ \hat{F} = \nabla m  \f$
           In the case of a shell it is
           \f$ \hat{F} = (\nabla m | \overline{R}_3) \f$
         */
        FieldMatrix<T,dimworld,dimworld> F;
        for (int i=0; i<dimworld; i++)
          for (int j=0; j<dim; j++)
            F[i][j] = deformationGradient[i][j];

        for (int i=0; i<dimworld; i++)
          for (int j=dim; j<dimworld; j++)
            F[i][j] = R[i][j];

        // U = R^T F
        for (int i=0; i<dimworld; i++)
          for (int j=0; j<dimworld; j++) {
            data_[i][j] = 0;
            for (int k=0; k<dimworld; k++)
              data_[i][j] += R[k][i] * F[k][j];
          }

      }

      /** \brief Compute strain from the deformation gradient and the microrotation
       *
       * \param R The microrotation
       */
      CosseratStrain(const FieldMatrix<T,dimworld,dim>& deformationGradient,
                     const FieldMatrix<T,dimworld,dimworld>& R)
      {
        /* Compute F, the deformation gradient.
           In the continuum case (domain dimension == world dimension) this is
           \f$ \hat{F} = \nabla m  \f$
           In the case of a shell it is
           \f$ \hat{F} = (\nabla m | \overline{R}_3) \f$
         */
        FieldMatrix<T,dimworld,dimworld> F;
        for (int i=0; i<dimworld; i++)
          for (int j=0; j<dim; j++)
            F[i][j] = deformationGradient[i][j];

        for (int i=0; i<dimworld; i++)
          for (int j=dim; j<dimworld; j++)
            F[i][j] = R[i][j];

        // U = R^T F
        for (int i=0; i<dimworld; i++)
          for (int j=0; j<dimworld; j++) {
            data_[i][j] = 0;
            for (int k=0; k<dimworld; k++)
              data_[i][j] += R[k][i] * F[k][j];
          }

      }

      T determinant() const
      {
        // If dim==2 we know that the last column is the identity matrix.
        // Hence the determinant is equal to the determinant of the
        // upper left 2x2 sub-block.
        if (dim==2)
          return data_[0][0]*data_[1][1] - data_[1][0]*data_[0][1];

        return data_.determinant();
      }

      /** \brief Temporary: get the raw matrix data */
      const FieldMatrix<T,dimworld,dimworld>& matrix() const
      {
        return data_;
      }

    private:
      FieldMatrix<T,dimworld,dimworld> data_;
    };

  }   // namespace GFE

}  // namespace Dune

#endif   // DUNE_GFE_COSSERATSTRAIN_HH
