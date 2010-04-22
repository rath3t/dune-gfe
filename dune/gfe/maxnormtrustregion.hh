#ifndef MAX_NORM_TRUST_REGION_HH
#define MAX_NORM_TRUST_REGION_HH

#include <vector>

#include <dune/solvers/boxconstraint.hh>

template <int blocksize>
class MaxNormTrustRegion
{
public:

    MaxNormTrustRegion(size_t size, double initialRadius) 
        : obstacles_(size)
    {
        set(initialRadius);
    }

    void set(double radius) {

        for (size_t i=0; i<obstacles_.size(); i++) {

            for (int k=0; k<blocksize; k++) {
                
                obstacles_[i].lower(k) = -radius;
                obstacles_[i].upper(k) =  radius;
                
            }
            
        }

    }

    double radius() const {
        assert(obstacles_.size()>0);
        assert(blocksize>0);
        return obstacles_[0].upper(0);
    }

    void scale(double factor) {

        for (size_t i=0; i<obstacles_.size(); i++) {
            
            for (int k=0; k<blocksize; k++) {
                
                obstacles_[i].lower(k) *= factor;
                obstacles_[i].upper(k) *= factor;
                
            }
            
        }

    }

    const std::vector<BoxConstraint<double,blocksize> >& obstacles() const {
        return obstacles_;
    }

private:

    std::vector<BoxConstraint<double,blocksize> > obstacles_;

};

#endif
