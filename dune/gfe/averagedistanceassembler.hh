#ifndef AVERAGE_DISTANCE_ASSEMBLER_HH
#define AVERAGE_DISTANCE_ASSEMBLER_HH

#include <vector>

#include "rotation.hh"

template <class TargetSpace>
class AverageDistanceAssembler
{
    static const int size = TargetSpace::EmbeddedTangentVector::size;

public:

    AverageDistanceAssembler(const std::vector<TargetSpace>& coefficients,
                             const std::vector<double>& weights)
        : coefficients_(coefficients),
          weights_(weights)
    {}

    double value(const TargetSpace& x) const {

        double result = 0;
        for (size_t i=0; i<coefficients_.size(); i++) {
            double dist = TargetSpace::distance(coefficients_[i], x);
            result += 0.5*weights_[i]*dist*dist;
        }

        return result;
    }

    void assembleGradient(const TargetSpace& x,
                          typename TargetSpace::EmbeddedTangentVector& gradient) const
    {
        gradient = 0;
        for (size_t i=0; i<coefficients_.size(); i++)
            gradient.axpy(0.5*weights_[i], 
                          TargetSpace::derivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i], x));
    }

    void assembleHessianApproximation(const TargetSpace& x,
                                      Dune::FieldMatrix<double,size,size>& matrix) const
    {
        for (int i=0; i<size; i++)
            for (int j=0; j<size; j++)
                matrix[i][j] = (i==j);
    }

    void assembleHessian(const TargetSpace& x,
                         Dune::FieldMatrix<double,size,size>& matrix) const
    {
        matrix = 0;
        for (size_t i=0; i<coefficients_.size(); i++)
            matrix.axpy(weights_[i], TargetSpace::secondDerivativeOfDistanceSquaredWRTSecondArgument(coefficients_[i], x));
    }

    const std::vector<TargetSpace> coefficients_;

    const std::vector<double> weights_;

};


template <>
class AverageDistanceAssembler<Rotation<3,double> >
{
    typedef Rotation<3,double> TargetSpace;

    static const int size = TargetSpace::TangentVector::size;

public:

    AverageDistanceAssembler(const std::vector<TargetSpace>& coefficients,
                             const std::vector<double>& weights)
        : coefficients_(coefficients),
          weights_(weights)
    {}

    double value(const TargetSpace& x) const {

        double result = 0;
        for (size_t i=0; i<coefficients_.size(); i++) {
            double dist = TargetSpace::distance(coefficients_[i], x);
            result += 0.5*weights_[i]*dist*dist;
        }

        return result;
    }

    void assembleGradient(const TargetSpace& x,
                          TargetSpace::TangentVector& gradient) const
    {
        DUNE_THROW(Dune::NotImplemented, "assembleGradient");
    }

    void assembleHessianApproximation(const TargetSpace& x,
                                      Dune::FieldMatrix<double,size,size>& matrix) const
    {
        DUNE_THROW(Dune::NotImplemented, "assembleHessianApproximation");
    }

    void assembleHessian(const TargetSpace& x,
                         Dune::FieldMatrix<double,size,size>& matrix) const
    {
        DUNE_THROW(Dune::NotImplemented, "assembleHessian");
    }

    const std::vector<TargetSpace> coefficients_;

    const std::vector<double> weights_;

};

#endif
