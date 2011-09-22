#ifndef AVERAGE_DISTANCE_ASSEMBLER_HH
#define AVERAGE_DISTANCE_ASSEMBLER_HH

#include <vector>

/** \tparam N Number of coefficients (i.e., simplex corners) */
template <class TargetSpace, int N>
class AverageDistanceAssembler
{
    static const int size         = TargetSpace::TangentVector::dimension;
    static const int embeddedSize = TargetSpace::EmbeddedTangentVector::dimension;

public:

    AverageDistanceAssembler(const Dune::array<TargetSpace,N>& coefficients,
                             const Dune::array<double,N>& weights) DUNE_DEPRECATED
        : coefficients_(coefficients.begin(), coefficients.end()),
          weights_(weights.begin(), weights.end())
    {}

    AverageDistanceAssembler(const std::vector<TargetSpace>& coefficients,
                             const std::vector<double>& weights)
        : coefficients_(coefficients),
          weights_(weights)
    {}

    double value(const TargetSpace& x) const {

        double result = 0;
        for (size_t i=0; i<coefficients_.size(); i++) {
            double dist = TargetSpace::distance(coefficients_[i], x);
            result += weights_[i]*dist*dist;
        }

        return result;
    }

    void assembleEmbeddedGradient(const TargetSpace& x,
                          typename TargetSpace::EmbeddedTangentVector& gradient) const
    {
        gradient = 0;
        for (size_t i=0; i<coefficients_.size(); i++)
            gradient.axpy(weights_[i], 
                          TargetSpace::derivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i], x));
    }

    void assembleGradient(const TargetSpace& x,
                          typename TargetSpace::TangentVector& gradient) const
    {
        typename TargetSpace::EmbeddedTangentVector embeddedGradient;
        assembleEmbeddedGradient(x,embeddedGradient);
        
        Dune::FieldMatrix<double,size,embeddedSize> orthonormalFrame = x.orthonormalFrame();
        orthonormalFrame.mv(embeddedGradient,gradient);
    }

    void assembleEmbeddedHessian(const TargetSpace& x,
                         Dune::FieldMatrix<double,embeddedSize,embeddedSize>& matrix) const
    {
        matrix = 0;
        for (size_t i=0; i<coefficients_.size(); i++)
            matrix.axpy(weights_[i], TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i], x));
    }

    void assembleHessian(const TargetSpace& x,
                         Dune::FieldMatrix<double,size,size>& matrix) const
    {
        Dune::FieldMatrix<double,embeddedSize,embeddedSize> embeddedHessian;
        assembleEmbeddedHessian(x,embeddedHessian);
        
        Dune::FieldMatrix<double,size,embeddedSize> frame = x.orthonormalFrame();
        
        // this is frame * embeddedHessian * frame^T
        for (int i=0; i<size; i++)
            for (int j=0; j<size; j++) {
                matrix[i][j] = 0;
                for (int k=0; k<embeddedSize; k++)
                    for (int l=0; l<embeddedSize; l++)
                        matrix[i][j] += frame[i][k]*embeddedHessian[k][l]*frame[j][l];
            }
        
    }

    const std::vector<TargetSpace> coefficients_;

    const std::vector<double> weights_;

};

#endif
