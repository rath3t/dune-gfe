#ifndef AVERAGE_DISTANCE_ASSEMBLER_HH
#define AVERAGE_DISTANCE_ASSEMBLER_HH

#include <vector>

#include "rotation.hh"

template <class TargetSpace>
class AverageDistanceAssembler
{};


template <>
class AverageDistanceAssembler<Rotation<3,double> >
{
    typedef Rotation<3,double> TargetSpace;

    static const int size = TargetSpace::TangentVector::size;

public:

    AverageDistanceAssembler(const std::vector<TargetSpace> coefficients,
                             const std::vector<double> weights)
        : coefficients_(coefficients),
          weights_(weights)
    {}

    double value(const TargetSpace& x) {

        double result = 0;
        for (size_t i=0; i<coefficients_.size(); i++) {
            double dist = TargetSpace::distance(coefficients_[i], x);
            result += 0.5*weights_[i]*dist*dist;
        }

        return result;
    }

    void assembleGradient(const TargetSpace& x,
                          TargetSpace::TangentVector& gradient)
    {
        DUNE_THROW(Dune::NotImplemented, "assembleGradient");
    }

    void assembleMatrix(const TargetSpace& x,
                        Dune::FieldMatrix<double,size,size>& matrix)
    {
        DUNE_THROW(Dune::NotImplemented, "assembleMatrix");
    }

    const std::vector<TargetSpace> coefficients_;

    const std::vector<double> weights_;

};

#endif
