#ifndef MAX_NORM_TRUST_REGION_HH
#define MAX_NORM_TRUST_REGION_HH

#include <vector>

#include <dune/solvers/common/boxconstraint.hh>

template <int blocksize, class field_type=double>
class MaxNormTrustRegion
{
public:

    MaxNormTrustRegion(size_t size, field_type initialRadius)
        : obstacles_(size)
    {
        set(initialRadius);
    }

    void set(field_type radius) {

        radius_ = radius;

        for (size_t i=0; i<obstacles_.size(); i++) {

            for (int k=0; k<blocksize; k++) {

                obstacles_[i].lower(k) = -radius;
                obstacles_[i].upper(k) =  radius;

            }

        }

    }

    field_type radius() const {
        return radius_;
    }

    void scale(field_type factor) {

        radius_ *= factor;

        for (size_t i=0; i<obstacles_.size(); i++) {

            for (int k=0; k<blocksize; k++) {

                obstacles_[i].lower(k) *= factor;
                obstacles_[i].upper(k) *= factor;

            }

        }

    }

    const std::vector<BoxConstraint<field_type,blocksize> >& obstacles() const {
        return obstacles_;
    }

private:

    std::vector<BoxConstraint<field_type,blocksize> > obstacles_;

    field_type radius_;

};

#endif
