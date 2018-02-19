#ifndef ROD_LOCAL_STIFFNESS_HH
#define ROD_LOCAL_STIFFNESS_HH

#include <array>

#include <dune/common/fmatrix.hh>
#include <dune/istl/matrix.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/functions/functionspacebases/pqknodalbasis.hh>

#include "localgeodesicfestiffness.hh"
#include "rigidbodymotion.hh"

template<class GridView, class RT>
class RodLocalStiffness
    : public LocalGeodesicFEStiffness<Dune::Functions::PQkNodalBasis<GridView,1>, RigidBodyMotion<RT,3> >
{
    typedef RigidBodyMotion<RT,3> TargetSpace;
    typedef Dune::Functions::PQkNodalBasis<GridView,1> Basis;

    // grid types
    typedef typename GridView::Grid::ctype DT;
    typedef typename GridView::template Codim<0>::Entity Entity;

    // some other sizes
    enum {dim=GridView::dimension};

    // Quadrature order used for the extension and shear energy
    enum {shearQuadOrder = 2};

    // Quadrature order used for the bending and torsion energy
    enum {bendingQuadOrder = 2};

public:

    /** \brief The stress-free configuration */
    std::vector<RigidBodyMotion<RT,3> > referenceConfiguration_;

public:

    //! Each block is x, y, theta in 2d, T (R^3 \times SO(3)) in 3d
    enum { blocksize = 6 };

    // define the number of components of your system, this is used outside
    // to allocate the correct size of (dense) blocks with a FieldMatrix
    enum {m=blocksize};

    // types for matrics, vectors and boundary conditions
    typedef Dune::FieldMatrix<RT,m,m> MBlockType; // one entry in the stiffness matrix
    typedef Dune::FieldVector<RT,m> VBlockType;   // one entry in the global vectors

    // /////////////////////////////////
    //   The material parameters
    // /////////////////////////////////

    /** \brief Material constants */
    std::array<double,3> K_;
    std::array<double,3> A_;

    GridView gridView_;

    //! Constructor
    RodLocalStiffness (const GridView& gridView,
                       const std::array<double,3>& K, const std::array<double,3>& A)
        : gridView_(gridView)
    {
        for (int i=0; i<3; i++) {
            K_[i] = K[i];
            A_[i] = A[i];
        }
    }

    /** \brief Constructor setting shape constants and material parameters
        \param A The rod section area
        \param J1, J2 The geometric moments (Flächenträgheitsmomente)
        \param E Young's modulus
        \param nu Poisson number
    */
    RodLocalStiffness (const GridView& gridView,
                       double A, double J1, double J2, double E, double nu)
        : gridView_(gridView)
    {
        // shear modulus
        double G = E/(2+2*nu);

        K_[0] = E * J1;
        K_[1] = E * J2;
        K_[2] = G * (J1 + J2);

        A_[0] = G * A;
        A_[1] = G * A;
        A_[2] = E * A;
    }



    void setReferenceConfiguration(const std::vector<RigidBodyMotion<RT,3> >& referenceConfiguration) {
        referenceConfiguration_ = referenceConfiguration;
    }

    /** \brief Local element energy for a P1 element */
    virtual RT energy (const Entity& e,
                       const std::array<RigidBodyMotion<RT,3>, dim+1>& localSolution) const;

    virtual RT energy (const typename Basis::LocalView& localView,
                       const std::vector<RigidBodyMotion<RT,3> >& localSolution) const
    {
        assert(localSolution.size()==2);
        std::array<RigidBodyMotion<RT,3>, 2> localSolutionArray = {localSolution[0], localSolution[1]};
        return energy(localView.element(),localSolutionArray);
    }

    /** \brief Assemble the element gradient of the energy functional */
    void assembleGradient(const Entity& element,
                          const std::vector<RigidBodyMotion<RT,3> >& solution,
                          std::vector<Dune::FieldVector<double,6> >& gradient) const;

    Dune::FieldVector<double, 6> getStrain(const std::vector<RigidBodyMotion<RT,3> >& localSolution,
                                           const Entity& element,
                                           const Dune::FieldVector<double,1>& pos) const;

protected:

    void getLocalReferenceConfiguration(const Entity& element,
                                        std::vector<RigidBodyMotion<RT,3> >& localReferenceConfiguration) const {

        unsigned int numOfBaseFct = element.subEntities(dim);
        localReferenceConfiguration.resize(numOfBaseFct);

        for (size_t i=0; i<numOfBaseFct; i++)
            localReferenceConfiguration[i] = referenceConfiguration_[gridView_.indexSet().subIndex(element,i,dim)];
    }

public:
    static void interpolationDerivative(const Rotation<RT,3>& q0, const Rotation<RT,3>& q1, double s,
                                        std::array<Quaternion<double>,6>& grad);

    static void interpolationVelocityDerivative(const Rotation<RT,3>& q0, const Rotation<RT,3>& q1, double s,
                                                double intervalLength, std::array<Quaternion<double>,6>& grad);

protected:
    template <class T>
    static Dune::FieldVector<T,3> darboux(const Rotation<T,3>& q, const Dune::FieldVector<T,4>& q_s)
    {
        Dune::FieldVector<double,3> u;  // The Darboux vector

        u[0] = 2 * (q.B(0) * q_s);
        u[1] = 2 * (q.B(1) * q_s);
        u[2] = 2 * (q.B(2) * q_s);

        return u;
    }

};

template <class GridType, class RT>
RT RodLocalStiffness<GridType, RT>::
energy(const Entity& element,
       const std::array<RigidBodyMotion<RT,3>, dim+1>& localSolution
       ) const
{
    RT energy = 0;

    std::vector<RigidBodyMotion<RT,3> > localReferenceConfiguration;
    getLocalReferenceConfiguration(element, localReferenceConfiguration);

    // ///////////////////////////////////////////////////////////////////////////////
    //   The following two loops are a reduced integration scheme.  We integrate
    //   the transverse shear and extensional energy with a first-order quadrature
    //   formula, even though it should be second order.  This prevents shear-locking.
    // ///////////////////////////////////////////////////////////////////////////////

    const Dune::QuadratureRule<double, 1>& shearingQuad
        = Dune::QuadratureRules<double, 1>::rule(element.type(), shearQuadOrder);

    // hack: convert from std::array to std::vector
    std::vector<RigidBodyMotion<RT,3> > localSolutionVector(localSolution.begin(), localSolution.end());

    for (size_t pt=0; pt<shearingQuad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<double,1>& quadPos = shearingQuad[pt].position();

        const double integrationElement = element.geometry().integrationElement(quadPos);

        double weight = shearingQuad[pt].weight() * integrationElement;

        Dune::FieldVector<double,6> strain = getStrain(localSolutionVector, element, quadPos);

        // The reference strain
        Dune::FieldVector<double,6> referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);

        for (int i=0; i<3; i++)
            energy += weight * 0.5 * A_[i] * (strain[i] - referenceStrain[i]) * (strain[i] - referenceStrain[i]);

    }

    // Get quadrature rule
    const Dune::QuadratureRule<double, 1>& bendingQuad
        = Dune::QuadratureRules<double, 1>::rule(element.type(), bendingQuadOrder);

    for (size_t pt=0; pt<bendingQuad.size(); pt++) {

        // Local position of the quadrature point
        const Dune::FieldVector<double,1>& quadPos = bendingQuad[pt].position();

        double weight = bendingQuad[pt].weight() * element.geometry().integrationElement(quadPos);

        Dune::FieldVector<double,6> strain = getStrain(localSolutionVector, element, quadPos);

        // The reference strain
        Dune::FieldVector<double,6> referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);

        // Part II: the bending and twisting energy
        for (int i=0; i<3; i++)
            energy += weight * 0.5 * K_[i] * (strain[i+3] - referenceStrain[i+3]) * (strain[i+3] - referenceStrain[i+3]);

    }

    return energy;
}


template <class GridType, class RT>
void RodLocalStiffness<GridType, RT>::
interpolationDerivative(const Rotation<RT,3>& q0, const Rotation<RT,3>& q1, double s,
                        std::array<Quaternion<double>,6>& grad)
{
    // Clear output array
    for (int i=0; i<6; i++)
        grad[i] = 0;

    // The derivatives with respect to w^0

    // Compute q_1^{-1}q_0
    Rotation<RT,3> q1InvQ0 = q1;
    q1InvQ0.invert();
    q1InvQ0 = q1InvQ0.mult(q0);

    {
    // Compute v = (1-s) \exp^{-1} ( q_1^{-1} q_0)
        Dune::FieldVector<RT,3> v = Rotation<RT,3>::expInv(q1InvQ0);
    v *= (1-s);

    Dune::FieldMatrix<RT,4,3> dExp_v = Rotation<RT,3>::Dexp(SkewMatrix<RT,3>(v));

    Dune::FieldMatrix<RT,3,4> dExpInv = Rotation<RT,3>::DexpInv(q1InvQ0);

    Dune::FieldMatrix<RT,4,4> mat(0);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            for (int k=0; k<3; k++)
                mat[i][j] += (1-s) * dExp_v[i][k] * dExpInv[k][j];

    for (int i=0; i<3; i++) {

        Quaternion<RT> dw;
        for (int j=0; j<4; j++)
            dw[j] = 0.5 * (i==j);  // dExp[j][i] at v=0

        dw = q1InvQ0.Quaternion<double>::mult(dw);

        mat.umv(dw,grad[i]);
        grad[i] = q1.Quaternion<double>::mult(grad[i]);

    }
    }


    // The derivatives with respect to w^1

    // Compute q_0^{-1}
    Rotation<RT,3> q0InvQ1 = q0;
    q0InvQ1.invert();
    q0InvQ1 = q0InvQ1.mult(q1);

    {
    // Compute v = s \exp^{-1} ( q_0^{-1} q_1)
        Dune::FieldVector<RT,3> v = Rotation<RT,3>::expInv(q0InvQ1);
    v *= s;

    Dune::FieldMatrix<RT,4,3> dExp_v = Rotation<RT,3>::Dexp(SkewMatrix<RT,3>(v));

    Dune::FieldMatrix<RT,3,4> dExpInv = Rotation<RT,3>::DexpInv(q0InvQ1);

    Dune::FieldMatrix<RT,4,4> mat(0);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            for (int k=0; k<3; k++)
                mat[i][j] += s * dExp_v[i][k] * dExpInv[k][j];

    for (int i=3; i<6; i++) {

        Quaternion<RT> dw;
        for (int j=0; j<4; j++)
            dw[j] = 0.5 * ((i-3)==j);  // dExp[j][i-3] at v=0

        dw = q0InvQ1.Quaternion<double>::mult(dw);

        mat.umv(dw,grad[i]);
        grad[i] = q0.Quaternion<double>::mult(grad[i]);

    }
    }
}



template <class GridType, class RT>
void RodLocalStiffness<GridType, RT>::
interpolationVelocityDerivative(const Rotation<RT,3>& q0, const Rotation<RT,3>& q1, double s,
                                double intervalLength, std::array<Quaternion<double>,6>& grad)
{
    // Clear output array
    for (int i=0; i<6; i++)
        grad[i] = 0;

    // Compute q_0^{-1}
    Rotation<RT,3> q0Inv = q0;
    q0Inv.invert();


    // Compute v = s \exp^{-1} ( q_0^{-1} q_1)
    Dune::FieldVector<RT,3> v = Rotation<RT,3>::expInv(q0Inv.mult(q1));
    v *= s/intervalLength;

    Dune::FieldMatrix<RT,4,3> dExp_v = Rotation<RT,3>::Dexp(SkewMatrix<RT,3>(v));

    std::array<Dune::FieldMatrix<RT,3,3>, 4> ddExp;
    Rotation<RT,3>::DDexp(v, ddExp);

    Dune::FieldMatrix<RT,3,4> dExpInv = Rotation<RT,3>::DexpInv(q0Inv.mult(q1));

    Dune::FieldMatrix<RT,4,4> mat(0);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            for (int k=0; k<3; k++)
                mat[i][j] += 1/intervalLength * dExp_v[i][k] * dExpInv[k][j];


    // /////////////////////////////////////////////////
    // The derivatives with respect to w^0
    // /////////////////////////////////////////////////
    for (int i=0; i<3; i++) {

        // \partial exp \partial w^1_j at 0
        Quaternion<RT> dw;
        for (int j=0; j<4; j++)
            dw[j] = 0.5*(i==j);  // dExp_v_0[j][i];

        // \xi = \exp^{-1} q_0^{-1} q_1
        Dune::FieldVector<RT,3> xi = Rotation<RT,3>::expInv(q0Inv.mult(q1));

        Quaternion<RT> addend0;
        addend0 = 0;
        dExp_v.umv(xi,addend0);
        addend0 = dw.mult(addend0);
        addend0 /= intervalLength;

        //  \parder{\xi}{w^1_j} = ...
        Quaternion<RT> dwConj = dw;
        dwConj.conjugate();
        //dwConj[3] -= 2 * dExp_v_0[3][i];   the last row of dExp_v_0 is zero
        dwConj = dwConj.mult(q0Inv.mult(q1));

        Dune::FieldVector<RT,3> dxi(0);
        Rotation<RT,3>::DexpInv(q0Inv.mult(q1)).umv(dwConj, dxi);

        Quaternion<RT> vHv;
        for (int j=0; j<4; j++) {
            vHv[j] = 0;
            // vHv[j] = dxi * DDexp * xi
            for (int k=0; k<3; k++)
                for (int l=0; l<3; l++)
                    vHv[j] += ddExp[j][k][l]*dxi[k]*xi[l];
        }

        vHv *= s/intervalLength/intervalLength;

        // Third addend
        mat.umv(dwConj,grad[i]);

        // add up
        grad[i] += addend0;
        grad[i] += vHv;

        grad[i] = q0.Quaternion<double>::mult(grad[i]);
    }


    // /////////////////////////////////////////////////
    // The derivatives with respect to w^1
    // /////////////////////////////////////////////////
    for (int i=3; i<6; i++) {

        // \partial exp \partial w^1_j at 0
        Quaternion<RT> dw;
        for (int j=0; j<4; j++)
            dw[j] = 0.5 * ((i-3)==j);  // dw[j] = dExp_v_0[j][i-3];

        // \xi = \exp^{-1} q_0^{-1} q_1
        Dune::FieldVector<RT,3> xi = Rotation<RT,3>::expInv(q0Inv.mult(q1));

        //  \parder{\xi}{w^1_j} = ...
        Dune::FieldVector<RT,3> dxi(0);
        dExpInv.umv(q0Inv.Quaternion<double>::mult(q1.Quaternion<double>::mult(dw)), dxi);

        Quaternion<RT> vHv;
        for (int j=0; j<4; j++) {
            // vHv[j] = dxi * DDexp * xi
            vHv[j] = 0;
            for (int k=0; k<3; k++)
                for (int l=0; l<3; l++)
                    vHv[j] += ddExp[j][k][l]*dxi[k]*xi[l];
        }

        vHv *= s/intervalLength/intervalLength;

        // ///////////////////////////////////
        //   second addend
        // ///////////////////////////////////


        dw = q0Inv.Quaternion<double>::mult(q1.Quaternion<double>::mult(dw));

        mat.umv(dw,grad[i]);
        grad[i] += vHv;

        grad[i] = q0.Quaternion<double>::mult(grad[i]);

    }

}

template <class GridType, class RT>
Dune::FieldVector<double, 6> RodLocalStiffness<GridType, RT>::
getStrain(const std::vector<RigidBodyMotion<RT,3> >& localSolution,
          const Entity& element,
          const Dune::FieldVector<double,1>& pos) const
{
    if (!element.isLeaf())
        DUNE_THROW(Dune::NotImplemented, "Only for leaf elements");

    assert(localSolution.size() == 2);

    // Strain defined on each element
    Dune::FieldVector<double, 6> strain(0);

    // Extract local solution on this element
    Dune::P1LocalFiniteElement<double,double,1> localFiniteElement;
    int numOfBaseFct = localFiniteElement.localCoefficients().size();

    const Dune::FieldMatrix<double,1,1>& inv = element.geometry().jacobianInverseTransposed(pos);

    // ///////////////////////////////////////
    //   Compute deformation gradient
    // ///////////////////////////////////////
    std::vector<Dune::FieldMatrix<double,1,1> > shapeGrad;

    localFiniteElement.localBasis().evaluateJacobian(pos, shapeGrad);

    for (int dof=0; dof<numOfBaseFct; dof++) {

        // multiply with jacobian inverse
        Dune::FieldVector<double,1> tmp(0);
        inv.umv(shapeGrad[dof][0], tmp);
        shapeGrad[dof] = tmp;

    }

    // //////////////////////////////////
    //   Interpolate
    // //////////////////////////////////

    Dune::FieldVector<double,3> r_s;
    for (int i=0; i<3; i++)
        r_s[i] = localSolution[0].r[i]*shapeGrad[0][0] + localSolution[1].r[i]*shapeGrad[1][0];

    // Interpolate the rotation at the quadrature point
    Rotation<RT,3> q = Rotation<RT,3>::interpolate(localSolution[0].q, localSolution[1].q, pos);

    // Get the derivative of the rotation at the quadrature point by interpolating in $H$
    Quaternion<double> q_s = Rotation<RT,3>::interpolateDerivative(localSolution[0].q, localSolution[1].q,
                                                                       pos);
    // Transformation from the reference element
    q_s *= inv[0][0];

    // /////////////////////////////////////////////
    //   Sum it all up
    // /////////////////////////////////////////////

    // Part I: the shearing and stretching strain
    strain[0] = r_s * q.director(0);    // shear strain
    strain[1] = r_s * q.director(1);    // shear strain
    strain[2] = r_s * q.director(2);    // stretching strain

    // Part II: the Darboux vector

    Dune::FieldVector<double,3> u = darboux(q, q_s);

    strain[3] = u[0];
    strain[4] = u[1];
    strain[5] = u[2];

    return strain;
}

template <class GridType, class RT>
void RodLocalStiffness<GridType, RT>::
assembleGradient(const Entity& element,
                 const std::vector<RigidBodyMotion<RT,3> >& solution,
                 std::vector<Dune::FieldVector<double,6> >& gradient) const
{
    using namespace Dune;

    std::vector<RigidBodyMotion<RT,3> > localReferenceConfiguration;
    getLocalReferenceConfiguration(element, localReferenceConfiguration);

    // Extract local solution on this element
    Dune::P1LocalFiniteElement<double,double,1> localFiniteElement;
    int numOfBaseFct = localFiniteElement.localCoefficients().size();

    // init
    gradient.resize(numOfBaseFct);
    for (size_t i=0; i<gradient.size(); i++)
        gradient[i] = 0;

    double intervalLength = element.geometry().corner(1)[0] - element.geometry().corner(0)[0];

    // ///////////////////////////////////////////////////////////////////////////////////
    //   Reduced integration to avoid locking:  assembly is split into a shear part
    //   and a bending part.  Different quadrature rules are used for the two parts.
    //   This avoids locking.
    // ///////////////////////////////////////////////////////////////////////////////////

    // Get quadrature rule
    const QuadratureRule<double, 1>& shearingQuad = QuadratureRules<double, 1>::rule(element.type(), shearQuadOrder);

    for (size_t pt=0; pt<shearingQuad.size(); pt++) {

        // Local position of the quadrature point
        const FieldVector<double,1>& quadPos = shearingQuad[pt].position();

        const FieldMatrix<double,1,1>& inv = element.geometry().jacobianInverseTransposed(quadPos);
        const double integrationElement = element.geometry().integrationElement(quadPos);

        double weight = shearingQuad[pt].weight() * integrationElement;

        // ///////////////////////////////////////
        //   Compute deformation gradient
        // ///////////////////////////////////////
        std::vector<Dune::FieldMatrix<double,1,1> > shapeGrad;
        localFiniteElement.localBasis().evaluateJacobian(quadPos, shapeGrad);

        for (int dof=0; dof<numOfBaseFct; dof++) {

            // multiply with jacobian inverse
            FieldVector<double,1> tmp(0);
            inv.umv(shapeGrad[dof][0], tmp);
            shapeGrad[dof] = tmp;

        }

        // //////////////////////////////////
        //   Interpolate
        // //////////////////////////////////

        FieldVector<double,3> r_s;
        for (int i=0; i<3; i++)
            r_s[i] = solution[0].r[i]*shapeGrad[0] + solution[1].r[i]*shapeGrad[1];

        // Interpolate current rotation at this quadrature point
        Rotation<RT,3> q = Rotation<RT,3>::interpolate(solution[0].q, solution[1].q,quadPos[0]);

        // The current strain
        FieldVector<double,blocksize> strain = getStrain(solution, element, quadPos);

        // The reference strain
        FieldVector<double,blocksize> referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);


        // dd_dvij[m][i][j] = \parder {(d_k)_i} {q}
        Tensor3<double,3 ,3, 4> dd_dq;
        q.getFirstDerivativesOfDirectors(dd_dq);

        // First derivatives of the position
        std::array<Quaternion<double>,6> dq_dwij;
        interpolationDerivative(solution[0].q, solution[1].q, quadPos, dq_dwij);

        // /////////////////////////////////////////////
        //   Sum it all up
        // /////////////////////////////////////////////

        for (int i=0; i<numOfBaseFct; i++) {

            // /////////////////////////////////////////////
            //   The translational part
            // /////////////////////////////////////////////

            // \partial \bar{W} / \partial r^i_j
            for (int j=0; j<3; j++) {

                for (int m=0; m<3; m++)
                    gradient[i][j] += weight
                        * (  A_[m] * (strain[m] - referenceStrain[m]) * shapeGrad[i] * q.director(m)[j]);

            }

            // \partial \bar{W}_v / \partial v^i_j
            for (int j=0; j<3; j++) {

                for (int m=0; m<3; m++) {
                    FieldVector<double,3> tmp(0);
                    dd_dq[m].umv(dq_dwij[3*i+j], tmp);
                    gradient[i][3+j] += weight
                        * A_[m] * (strain[m] - referenceStrain[m]) * (r_s*tmp);
                }
            }

        }

    }

    // /////////////////////////////////////////////////////
    //   Now: the bending/torsion part
    // /////////////////////////////////////////////////////

    // Get quadrature rule
    const QuadratureRule<double, 1>& bendingQuad = QuadratureRules<double, 1>::rule(element.type(), bendingQuadOrder);

    for (int pt=0; pt<bendingQuad.size(); pt++) {

        // Local position of the quadrature point
        const FieldVector<double,1>& quadPos = bendingQuad[pt].position();

        const FieldMatrix<double,1,1>& inv = element.geometry().jacobianInverseTransposed(quadPos);
        const double integrationElement = element.geometry().integrationElement(quadPos);

        double weight = bendingQuad[pt].weight() * integrationElement;

        // Interpolate current rotation at this quadrature point
        Rotation<RT,3> q = Rotation<RT,3>::interpolate(solution[0].q, solution[1].q,quadPos[0]);

        // Get the derivative of the rotation at the quadrature point by interpolating in $H$
        Quaternion<double> q_s = Rotation<RT,3>::interpolateDerivative(solution[0].q, solution[1].q,
                                                                           quadPos);
        // Transformation from the reference element
        q_s *= inv[0][0];


        // The current strain
        FieldVector<double,blocksize> strain = getStrain(solution, element, quadPos);

        // The reference strain
        FieldVector<double,blocksize> referenceStrain = getStrain(localReferenceConfiguration, element, quadPos);

        // First derivatives of the position
        std::array<Quaternion<double>,6> dq_dwij;
        interpolationDerivative(solution[0].q, solution[1].q, quadPos, dq_dwij);

        std::array<Quaternion<double>,6> dq_ds_dwij;
        interpolationVelocityDerivative(solution[0].q, solution[1].q, quadPos[0]*intervalLength, intervalLength,
                                        dq_ds_dwij);

        // /////////////////////////////////////////////
        //   Sum it all up
        // /////////////////////////////////////////////

        for (int i=0; i<numOfBaseFct; i++) {

            // /////////////////////////////////////////////
            //   The rotational part
            // /////////////////////////////////////////////

            // \partial \bar{W}_v / \partial v^i_j
            for (int j=0; j<3; j++) {

                for (int m=0; m<3; m++) {

                    // Compute derivative of the strain
                    /** \todo Is this formula correct?  It seems strange to call
                        B(m) for a _derivative_ of a rotation */
                    double du_dvij_m = 2 * (dq_dwij[i*3+j].B(m) * q_s)
                        + 2* ( q.B(m) * dq_ds_dwij[i*3+j]);

                    // Sum it up
                    gradient[i][3+j] += weight * K_[m]
                        * (strain[m+3]-referenceStrain[m+3]) * du_dvij_m;

                }

            }

        }

    }
}


#endif

