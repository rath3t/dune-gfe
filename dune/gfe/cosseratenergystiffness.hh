#ifndef COSSERAT_ENERGY_LOCAL_STIFFNESS_HH
#define COSSERAT_ENERGY_LOCAL_STIFFNESS_HH

#include <dune/common/fmatrix.hh>
#include <dune/common/parametertree.hh>
#include <dune/grid/common/quadraturerules.hh>

#include <dune/fufem/functions/virtualgridfunction.hh>

#include "localgeodesicfestiffness.hh"
#include "localgeodesicfefunction.hh"
#include <dune/gfe/rigidbodymotion.hh>


template<class GridView, class LocalFiniteElement, int dim>
class CosseratEnergyLocalStiffness 
    : public LocalGeodesicFEStiffness<GridView,LocalFiniteElement,RigidBodyMotion<dim> >
{
    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef RigidBodyMotion<dim> TargetSpace;
    typedef typename TargetSpace::ctype RT;
    typedef typename GridView::template Codim<0>::Entity Entity;
    
    // some other sizes
    enum {gridDim=GridView::dimension};
    
    
    /** \brief Compute the symmetric part of a matrix A, i.e. \f$ \frac 12 (A + A^T) \f$ */
    static Dune::FieldMatrix<double,dim,dim> sym(const Dune::FieldMatrix<double,dim,dim>& A)
    {
        Dune::FieldMatrix<double,dim,dim> result;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++)
                result[i][j] = 0.5 * (A[i][j] + A[j][i]);
        return result;
    }

    /** \brief Compute the antisymmetric part of a matrix A, i.e. \f$ \frac 12 (A - A^T) \f$ */
    static Dune::FieldMatrix<double,dim,dim> skew(const Dune::FieldMatrix<double,dim,dim>& A)
    {
        Dune::FieldMatrix<double,dim,dim> result;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++)
                result[i][j] = 0.5 * (A[i][j] - A[j][i]);
        return result;
    }
    
    /** \brief Return the square of the trace of a matrix */
    static double traceSquared(const Dune::FieldMatrix<double,dim,dim>& A)
    {
        double trace = 0;
        for (int i=0; i<dim; i++)
            trace += A[i][i];
        return trace*trace;
    }

    /** \brief Compute the (row-wise) curl of a matrix R \f$ 
        \param DR The partial derivatives of the matrix R
     */
    static Dune::FieldMatrix<double,dim,dim> curl(const Tensor3<double,dim,dim,dim>& DR)
    {
        Dune::FieldMatrix<double,dim,dim> result;
        
        for (int i=0; i<dim; i++) {
            result[i][0] = DR[i][2][1] - DR[i][1][2];
            result[i][1] = DR[i][0][2] - DR[i][2][0];
            result[i][2] = DR[i][1][0] - DR[i][0][1];
        }
            
        return result;
    }

public:  // for testing
    /** \brief Compute the derivative of the rotation, but wrt matrix coordinates */
    static void computeDR(const RigidBodyMotion<3>& value, 
                          const Dune::FieldMatrix<double,7,gridDim>& derivative,
                          Tensor3<double,3,3,3>& DR)
    {
        // The LocalGFEFunction class gives us the derivatives of the orientation variable,
        // but as a map into quaternion space.  To obtain matrix coordinates we use the
        // chain rule, which means that we have to multiply the given derivative with
        // the derivative of the embedding of the unit quaternion into the space of 3x3 matrices.
        // This second derivative is almost given by the method getFirstDerivativesOfDirectors.
        // However, since the directors of a given unit quaternion are the _columns_ of the
        // corresponding orthogonal matrix, we need to invert the i and j indices
        //
        // So, if I am not mistaken, DR[i][j][k] contains \partial M_ij / \partial k
        Tensor3<double,3 , 3, 4> dd_dq;
        value.q.getFirstDerivativesOfDirectors(dd_dq);
        
        DR = 0;
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                for (int k=0; k<gridDim; k++)
                    for (int l=0; l<4; l++)
                        DR[i][j][k] += dd_dq[j][i][l] * derivative[l+3][k];

    }

public:
    
    /** \brief Constructor with a set of material parameters
     * \param parameters The material parameters
     */
    CosseratEnergyLocalStiffness(const Dune::ParameterTree& parameters,
                                 const BoundaryPatch<GridView>* neumannBoundary,
                                 const Dune::VirtualFunction<Dune::FieldVector<double,gridDim>, Dune::FieldVector<double,3> >* neumannFunction)
    : neumannBoundary_(neumannBoundary),
      neumannFunction_(neumannFunction)
    {
        // The shell thickness
        thickness_ = parameters.template get<double>("thickness");
    
        // Lame constants
        mu_ = parameters.template get<double>("mu");
        lambda_ = parameters.template get<double>("lambda");

        // Cosserat couple modulus, preferably 0
        mu_c_ = parameters.template get<double>("mu_c");
    
        // Length scale parameter
        L_c_ = parameters.template get<double>("L_c");
    
        // Curvature exponent
        q_ = parameters.template get<double>("q");
    }

    /** \brief Assemble the energy for a single element */
    RT energy (const Entity& e,
               const LocalFiniteElement& localFiniteElement,
               const std::vector<TargetSpace>& localSolution) const;
               
    RT quadraticMembraneEnergy(const Dune::FieldMatrix<double,3,3>& U) const
    {
        Dune::FieldMatrix<double,3,3> UMinus1 = U;
        for (int i=0; i<dim; i++)
            UMinus1[i][i] -= 1;
        
        return mu_ * sym(UMinus1).frobenius_norm2()
                + mu_c_ * skew(UMinus1).frobenius_norm2()
                + (mu_*lambda_)/(2*mu_ + lambda_) * traceSquared(sym(UMinus1));
    }

    RT curvatureEnergy(const Tensor3<double,3,3,3>& DR) const
    {
        return mu_ * std::pow(L_c_ * curl(DR).frobenius_norm(),q_);
    }

    RT bendingEnergy(const Dune::FieldMatrix<double,dim,dim>& R, const Tensor3<double,3,3,3>& DR) const
    {
        // The derivative of the third director
        /** \brief There is no real need to have DR3 as a separate object */
        Dune::FieldMatrix<double,3,3> DR3;
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                DR3[i][j] = DR[i][2][j];
            
        // left-multiply with R^T
        Dune::FieldMatrix<double,3,3> RT_DR3;
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++) {
                RT_DR3[i][j] = 0;
                for (int k=0; k<3; k++)
                    RT_DR3[i][j] += R[k][i] * DR3[k][j];
            }
                
            
            
        return mu_ * sym(RT_DR3).frobenius_norm2()
               + mu_c_ * skew(RT_DR3).frobenius_norm2()
               /** \todo Is this sym correct?  It is in the paper, but not in the notes */
               + mu_*lambda_/(2*mu_+lambda_) * traceSquared(sym(RT_DR3));
    }

    /** \brief The shell thickness */
    double thickness_;
    
    /** \brief Lame constants */
    double mu_, lambda_;

    /** \brief Cosserat couple modulus, preferably 0 */
    double mu_c_;
    
    /** \brief Length scale parameter */
    double L_c_;
    
    /** \brief Curvature exponent */
    double q_;
    
    /** \brief The Neumann boundary */
    const BoundaryPatch<GridView>* neumannBoundary_;
    
    /** \brief The function implementing the Neumann data */
    const Dune::VirtualFunction<Dune::FieldVector<double,gridDim>, Dune::FieldVector<double,3> >* neumannFunction_;
};

template <class GridView, class LocalFiniteElement, int dim>
typename CosseratEnergyLocalStiffness<GridView,LocalFiniteElement,dim>::RT
CosseratEnergyLocalStiffness<GridView,LocalFiniteElement,dim>::
energy(const Entity& element,
       const LocalFiniteElement& localFiniteElement,
       const std::vector<RigidBodyMotion<dim> >& localSolution) const
{
    RT energy = 0;

    LocalGeodesicFEFunction<gridDim, double, LocalFiniteElement, TargetSpace> localGeodesicFEFunction(localFiniteElement,
                                                                                                      localSolution);

    int quadOrder = (element.type().isSimplex()) ? localFiniteElement.localBasis().order()
                                                 : localFiniteElement.localBasis().order() * gridDim;

    const Dune::QuadratureRule<double, gridDim>& quad 
        = Dune::QuadratureRules<double, gridDim>::rule(element.type(), quadOrder);
    
    for (size_t pt=0; pt<quad.size(); pt++) {
        
        // Local position of the quadrature point
        const Dune::FieldVector<double,gridDim>& quadPos = quad[pt].position();
        
        const double integrationElement = element.geometry().integrationElement(quadPos);

        const Dune::FieldMatrix<double,gridDim,gridDim>& jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(quadPos);
        
        double weight = quad[pt].weight() * integrationElement;
        
        // The value of the local function
        RigidBodyMotion<dim> value = localGeodesicFEFunction.evaluate(quadPos);

        // The derivative of the local function defined on the reference element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::dimension, gridDim> referenceDerivative = localGeodesicFEFunction.evaluateDerivative(quadPos);

        // The derivative of the function defined on the actual element
        Dune::FieldMatrix<double, TargetSpace::EmbeddedTangentVector::dimension, gridDim> derivative(0);

        for (size_t comp=0; comp<referenceDerivative.N(); comp++)
            jacobianInverseTransposed.umv(referenceDerivative[comp], derivative[comp]);
        
        /////////////////////////////////////////////////////////
        // compute U, the Cosserat strain
        /////////////////////////////////////////////////////////
        dune_static_assert(dim>=gridDim, "Codim of the grid must be nonnegative");
        
        //
        Dune::FieldMatrix<double,dim,dim> R;
        value.q.matrix(R);
        
        /* Compute F, the deformation gradient.
           In the continuum case this is
           \f$ \hat{F} = \nabla m  \f$
           In the case of a shell it is
          \f$ \hat{F} = (\nabla m | \overline{R}_3) \f$
        */
        Dune::FieldMatrix<double,dim,dim> F;
        for (int i=0; i<dim; i++)
            for (int j=0; j<gridDim; j++)
                F[i][j] = derivative[i][j];
            
        for (int i=0; i<dim; i++)
            for (int j=gridDim; j<dim; j++)
                F[i][j] = R[i][j];
        
        // U = R^T F
        Dune::FieldMatrix<double,dim,dim> U;
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++) {
                U[i][j] = 0;
                for (int k=0; k<dim; k++)
                    U[i][j] += R[k][i] * F[k][j];
            }
            
        //////////////////////////////////////////////////////////
        //  Compute the derivative of the rotation
        //  Note: we need it in matrix coordinates
        //////////////////////////////////////////////////////////
                
        Tensor3<double,3,3,3> DR;
        computeDR(value, derivative, DR);
        
        // Add the local energy density
        if (gridDim==2) {
            energy += weight * thickness_ * quadraticMembraneEnergy(U);
            energy += weight * thickness_ * curvatureEnergy(DR);
            energy += weight * std::pow(thickness_,3) / 12.0 * bendingEnergy(R,DR);
        } else if (gridDim==3) {
            energy += weight * quadraticMembraneEnergy(U);
            energy += weight * curvatureEnergy(DR);
        } else
            DUNE_THROW(Dune::NotImplemented, "CosseratEnergyStiffness for 1d grids");

    }
    
    
    //////////////////////////////////////////////////////////////////////////////
    //   Assemble boundary contributions
    //////////////////////////////////////////////////////////////////////////////
    
    assert(neumannFunction_);

    for (typename Entity::LeafIntersectionIterator it = element.ileafbegin(); it != element.ileafend(); ++it) {
        
        if (not neumannBoundary_ or not neumannBoundary_->contains(*it))
            continue;
        
        const Dune::QuadratureRule<double, gridDim-1>& quad 
            = Dune::QuadratureRules<double, gridDim-1>::rule(it->type(), quadOrder);
    
        for (size_t pt=0; pt<quad.size(); pt++) {
        
            // Local position of the quadrature point
            const Dune::FieldVector<double,gridDim>& quadPos = it->geometryInInside().global(quad[pt].position());
        
            const double integrationElement = it->geometry().integrationElement(quad[pt].position());

            // The value of the local function
            RigidBodyMotion<dim> value = localGeodesicFEFunction.evaluate(quadPos);

            // Value of the Neumann data at the current position
            Dune::FieldVector<double,3> neumannValue;
            
            if (dynamic_cast<const VirtualGridViewFunction<GridView,Dune::FieldVector<double,3> >*>(neumannFunction_))
                dynamic_cast<const VirtualGridViewFunction<GridView,Dune::FieldVector<double,3> >*>(neumannFunction_)->evaluateLocal(element, quadPos, neumannValue);
            else
                neumannFunction_->evaluate(it->geometry().global(quad[pt].position()), neumannValue);

            // Constant Neumann force in z-direction
            energy += thickness_ * (neumannValue * value.r) * quad[pt].weight() * integrationElement;
            
        }
        
    }
    
    return energy;
}

#endif

