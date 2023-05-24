#ifndef DUNE_GFE_SURFACECOSSERATSTRESSASSEMBLER_HH
#define DUNE_GFE_SURFACECOSSERATSTRESSASSEMBLER_HH

#include <dune/fufem/boundarypatch.hh>

#include <dune/gfe/linearalgebra.hh>
#include <dune/gfe/localgeodesicfefunction.hh>
#include <dune/gfe/spaces/rotation.hh>

#include <dune/matrix-vector/transpose.hh>

namespace Dune::GFE {
  /** \brief An assembler that can calculate the norms of specific stress tensors for each element for an output by film-on-substrate

    \tparam BasisOrderD Basis used for the displacement
    \tparam BasisOrderR Basis used for the rotation
    \tparam TargetSpaceD Target space for the Displacement
    \tparam TargetSpaceR Target space for the Rotation
  */
  template <class BasisOrderD, class BasisOrderR, class TargetSpaceD, class TargetSpaceR>
  class SurfaceCosseratStressAssembler
  {
    public:
      const static int dim = TargetSpaceD::dimension;
      using GridView = typename BasisOrderD::GridView;
      using VectorD = std::vector<TargetSpaceD>;
      using VectorR = std::vector<TargetSpaceR>;

      BasisOrderD basisOrderD_;
      BasisOrderR basisOrderR_;

      SurfaceCosseratStressAssembler(const BasisOrderD basisOrderD,
                                     const BasisOrderR basisOrderR)
      : basisOrderD_(basisOrderD),
        basisOrderR_(basisOrderR)
      {}

      /** \brief Calculate the norm of the 1st-Piola-Kirchhoff-Stress-Tensor and the Cauchy-Stress-Tensor for each element

          The 1st-Piola-Kirchhoff-Stress-Tensor is the derivative of the energy density with respect to the deformation gradient
          - Calculate the deformation gradient for each element using the basis functions and
            their gradients; then add them up using the localConfiguration
          - Evaluate the deformation gradient at each quadrature point using the respective quadrature rule with the given order
          - Evaluate the density function and tape the evaluation - then use ADOLC to evaluate the derivative (∂/∂F) W(F)
          - The derivative is then a dim x dim matrix
          - Then calculate the final stressTensor of the element by averagin over the quadrature points using the quadrature
            weights and the reference element volume
        \param x Coefficient vector for the displacement
        \param elasticDensity Energy density function
        \param int Order of the quadrature rule
        \param stressSubstrate1stPiolaKirchhoffTensor Vector containing the the 1st-Piola-Kirchhoff-Stress-Tensor for each element
        \param stressSubstrateCauchyTensor Vector containing the Cauchy-Stress-Tensor for each element
     */
      template <class Density>
      void assembleSubstrateStress(
        const VectorD x,
        const Density* elasticDensity,
        const int quadOrder,
        std::vector<FieldMatrix<double,dim,dim>>& stressSubstrate1stPiolaKirchhoffTensor,
        std::vector<FieldMatrix<double,dim,dim>>& stressSubstrateCauchyTensor)
      {

        std::cout << "Calculating the Frobenius norm of the 1st-Piola-Kirchhoff-Stress-Tensor ( (∂/∂F) W(F) )" << std::endl
                  << "and the Frobenius norm of the Cauchy-Stress-Tensor (1/det(F) * (∂/∂F) W(F) * F^T) of the substrate..." << std::endl;

        auto xFlat = Functions::istlVectorBackend(x);
        static constexpr auto partitionType = Partitions::interiorBorder;
        MultipleCodimMultipleGeomTypeMapper<GridView> elementMapper(basisOrderD_.gridView(),mcmgElementLayout());
        stressSubstrate1stPiolaKirchhoffTensor.resize(elementMapper.size());
        stressSubstrateCauchyTensor.resize(elementMapper.size());

        for (const auto& element : elements(basisOrderD_.gridView(), partitionType))
        {
          auto localViewOrderD = basisOrderD_.localView();
          localViewOrderD.bind(element);
          size_t nDofsOrderD = localViewOrderD.tree().size();

          // Extract values at this element
          std::vector<double> localConfiguration(nDofsOrderD);
          for (size_t i=0; i<nDofsOrderD; i++)
            localConfiguration[i] = xFlat[localViewOrderD.index(i)]; //localViewOrderD.index(i) is a multi-index

          //Store the reference gradient and the gradients for this element
          const auto& lFEOrderD = localViewOrderD.tree().child(0).finiteElement();
          std::vector<FieldMatrix<double,1,dim> > referenceGradients(lFEOrderD.size());
          std::vector<FieldMatrix<double,1,dim> > gradients(lFEOrderD.size());

          auto evaluateAtPoint = [&](FieldVector<double,3> pointGlobal, FieldVector<double,3> pointLocal) -> std::vector<FieldMatrix<double,dim,dim>>{
            std::vector<FieldMatrix<double,dim,dim>> stressTensors(2);

            const auto jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(pointLocal);

            // Get gradients of shape functions
            lFEOrderD.localBasis().evaluateJacobian(pointLocal, referenceGradients);

            // Compute gradients of Base functions
            for (size_t i=0; i<gradients.size(); ++i)
              gradients[i] = referenceGradients[i] * transpose(jacobianInverseTransposed);

            // Deformation gradient in vector form
            size_t nDoubles = dim*dim;
            std::vector<double> deformationGradientFlat(nDoubles);
            for (size_t i=0; i<nDoubles; i++)
              deformationGradientFlat[i] = 0;
            for (size_t i=0; i<gradients.size(); i++)
              for (size_t j=0; j<dim; j++)
                for (size_t k=0; k<dim; k++)
                  deformationGradientFlat[dim*j + k] += localConfiguration[ localViewOrderD.tree().child(j).localIndex(i)]*gradients[i][0][k];

            double pureDensity = 0;

            trace_on(0);

            FieldMatrix<adouble,dim,dim> deformationGradient(0);
            for (size_t j=0; j<dim; j++)
              for (size_t k=0; k<dim; k++)
                deformationGradient[j][k] <<= deformationGradientFlat[dim*j + k];

            // Tape the actual calculation
            adouble density = 0;
            try {
              density = (*elasticDensity)(pointGlobal, deformationGradient);
            } catch (Exception &e) {
              trace_off(0);
              throw e;
            }
            density >>= pureDensity;

            trace_off(0);

            // Compute the actual gradient
            std::vector<double> localStressFlat(nDoubles);
            gradient(0,nDoubles,deformationGradientFlat.data(),localStressFlat.data());

            FieldMatrix<double,dim,dim> localStress(0);
            for (size_t j=0; j<dim; j++)
              for (size_t k=0; k<dim; k++)
                localStress[j][k] = localStressFlat[dim*j + k];

            stressTensors[0] = localStress; // 1st-Piola-Kirchhoff-Stress-Tensor

            FieldMatrix<double,dim,dim> deformationGradientTransposed(0);
            for (size_t j=0; j<dim; j++)
              for (size_t k=0; k<dim; k++)
                deformationGradient[j][k] >>= deformationGradientTransposed[k][j];

            localStress /= deformationGradientTransposed.determinant();
            localStress = localStress * deformationGradientTransposed;
            stressTensors[1] = localStress; // Cauchy-Stress-Tensor

            return stressTensors;
          };

          //Call evaluateAtPoint for all points in the quadrature rule
          const auto& quad = Dune::QuadratureRules<double, dim>::rule(element.type(), quadOrder);
          stressSubstrate1stPiolaKirchhoffTensor[elementMapper.index(element)] = 0;
          stressSubstrateCauchyTensor[elementMapper.index(element)] = 0;

          for (size_t pt=0; pt<quad.size(); pt++) {
            auto pointLocal = quad[pt].position();
            auto pointGlobal = element.geometry().global(pointLocal);
            auto stressTensors = evaluateAtPoint(pointGlobal, pointLocal);
            stressSubstrate1stPiolaKirchhoffTensor[elementMapper.index(element)] += stressTensors[0] * quad[pt].weight()/referenceElement(element).volume();
            stressSubstrateCauchyTensor[elementMapper.index(element)] += stressTensors[1] * quad[pt].weight()/referenceElement(element).volume();
          }
        }
      }
      /** \brief  Calculate the norm of the Biot-Type-Stress-Tensor of the shell

          The formula for Biot-Type-Stress-Tensor of the Cosserat shell given by (4.11) in 
          Ghiba, Bîrsan, Lewintan, Neff, March 2020: "The isotropic Cosserat shell model including terms up to $O(h^5)$. Part I: Derivation in matrix notation"
        \param rot Coefficient vector for the rotation
        \param x Coefficient vector for the displacement
        \param xInitial Coefficient vector for the stress-free configuration of the shell, used to calculate nablaTheta
        \param lameF Function assigning the Lamé parameters to a given point
        \param mu_c Cosserat couple modulus
        \param shellBoundary BoundaryPatch containing the elements that actually belong to the shell
        \param order Order of the quadrature rule
        \param stressShellBiotTensor Vector containing the Biot-Stress-Tensor for each element
      */
      void assembleShellStress(
        const VectorR rot,
        const VectorD x,
        const VectorD xInitial,
        const std::function<Dune::FieldVector<double,2>(Dune::FieldVector<double,dim>)> lameF,
        const double mu_c,
        const BoundaryPatch<GridView> shellBoundary,
        const int quadOrder,
        std::vector<FieldMatrix<double,dim,dim>>& stressShellBiotTensor)
      {
        std::cout << "Calculating the Frobenius norm of the Biot-Type-Stress-Tensor of the shell..." << std::endl;
        auto xFlat = Functions::istlVectorBackend(x);
        auto xInitialFlat = Functions::istlVectorBackend(xInitial);

        MultipleCodimMultipleGeomTypeMapper<GridView> elementMapper(basisOrderD_.gridView(),mcmgElementLayout());
        stressShellBiotTensor.resize(elementMapper.size());

        static constexpr auto partitionType = Partitions::interiorBorder;
        for (const auto& element : elements(basisOrderD_.gridView(), partitionType))
        {
          stressShellBiotTensor[elementMapper.index(element)] = 0;

          int intersectionCounter = 0;
          for (auto&& it : intersections(shellBoundary.gridView(), element)) {
            FieldMatrix<double,dim,dim> stressTensorThisIntersection(0);

            //Continue if the element does not intersect with the shell boundary
            if (not shellBoundary.contains(it))
              continue;

            //LocalView for the basisOrderD_
            auto localViewOrderD = basisOrderD_.localView();
            localViewOrderD.bind(element);
            size_t nDofsOrderD = localViewOrderD.tree().size();

            // Extract local configuration at this element
            std::vector<double> localConfiguration(nDofsOrderD);
            std::vector<double> localConfigurationInitial(nDofsOrderD);
            for (size_t i=0; i<nDofsOrderD; i++) {
              localConfiguration[i] = xFlat[localViewOrderD.index(i)];
              localConfigurationInitial[i] = xInitialFlat[localViewOrderD.index(i)];
            }

            const auto& lFEOrderD = localViewOrderD.tree().child(0).finiteElement();
            //Store the reference gradient and the gradients for *this element*
            std::vector<FieldMatrix<double,1,dim> > referenceGradients(lFEOrderD.size());
            std::vector<FieldMatrix<double,1,dim> > gradients(lFEOrderD.size());

            //LocalView for the basisOrderR_
            auto localViewOrderR = basisOrderR_.localView();
            localViewOrderR.bind(element);
            const auto& lFEOrderR = localViewOrderR.tree().child(0).finiteElement();
            VectorR localConfigurationRot(lFEOrderR.size());
            for (std::size_t i=0; i<localConfigurationRot.size(); i++)
              localConfigurationRot[i] = rot[localViewOrderR.index(i)[0]];//localViewOrderR.index(i) is a multiindex, its first entry is the actual index
            typedef LocalGeodesicFEFunction<dim, double, decltype(lFEOrderR), TargetSpaceR> LocalGFEFunctionType;
            LocalGFEFunctionType localGeodesicFEFunction(lFEOrderR,localConfigurationRot);
            
            auto evaluateAtPoint = [&](FieldVector<double,3> pointGlobal, FieldVector<double,3> pointLocal3d) -> FieldMatrix<double,dim,dim>{
              Dune::FieldMatrix<double,dim,dim> nablaTheta;

              const auto jacobianInverseTransposed = element.geometry().jacobianInverseTransposed(pointLocal3d);

              // Get gradients of shape functions
              lFEOrderD.localBasis().evaluateJacobian(pointLocal3d, referenceGradients);

              // Compute gradients of Base functions at this element
              for (size_t i=0; i<gradients.size(); i++)
                gradients[i] = referenceGradients[i] * transpose(jacobianInverseTransposed);

              // Deformation gradient - call this U_es_minus_Id already
              FieldMatrix<double,dim,dim> U_es_minus_Id(0);
              for (size_t i=0; i<gradients.size(); i++)
                for (size_t j=0; j<dim; j++){
                  U_es_minus_Id[j].axpy(localConfiguration[ localViewOrderD.tree().child(j).localIndex(i)],gradients[i][0]);
                  nablaTheta[j].axpy(localConfigurationInitial[ localViewOrderD.tree().child(j).localIndex(i)],gradients[i][0]);
                }

              TargetSpaceR value = localGeodesicFEFunction.evaluate(pointLocal3d);
              FieldMatrix<double,dim,dim> rotationMatrix(0);
              FieldMatrix<double,dim,dim> rotationMatrixTransposed(0);
              value.matrix(rotationMatrix);

              MatrixVector::transpose(rotationMatrix, rotationMatrixTransposed);

              U_es_minus_Id.leftmultiply(rotationMatrixTransposed); // Attention: The rotation here is already is Q_e, we don't have to multiply with Q_0!!!

              nablaTheta.invert();
              U_es_minus_Id.rightmultiply(nablaTheta);

              for (size_t j = 0; j < dim; j++)
                U_es_minus_Id[j][j] -= 1.0;

              auto lameConstants = lameF(pointGlobal);
              double mu = lameConstants[0];
              double lambda = lameConstants[1];

              FieldMatrix<double,dim,dim> localStressShell(0);
              localStressShell = 2*mu*GFE::sym(U_es_minus_Id) + 2*mu_c*GFE::skew(U_es_minus_Id);
              for (size_t j = 0; j < dim; j++)
                localStressShell[j][j] += lambda*GFE::trace(GFE::sym(U_es_minus_Id));
              return localStressShell;
            };

            //Call evaluateAtPoint for all points in the quadrature rule
            const auto& quad = Dune::QuadratureRules<double, dim-1>::rule(it.type(), quadOrder); //Quad rule on the boundary 

            for (size_t pt=0; pt<quad.size(); pt++) {
              auto pointLocal2d = quad[pt].position();
              auto pointLocal3d = it.geometryInInside().global(pointLocal2d);
              auto pointGlobal = it.geometry().global(pointLocal2d);

              auto stressTensor = evaluateAtPoint(pointGlobal, pointLocal3d);
              stressTensorThisIntersection += stressTensor * quad[pt].weight()/referenceElement(it.inside()).volume();
            }
            stressShellBiotTensor[elementMapper.index(element)] += stressTensorThisIntersection;
            intersectionCounter++;
          }
          if (intersectionCounter >= 1)
            stressShellBiotTensor[elementMapper.index(element)] /= intersectionCounter;
        }
      }
  };
}
#endif
