#ifndef MAX_NORM_TRUST_REGION_HH
#define MAX_NORM_TRUST_REGION_HH

#include <vector>

#include <dune/solvers/common/boxconstraint.hh>


namespace Dune::GFE
{

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

  /** \brief Set trust region radius with a separate scaling for each vector block component
   */
  void set(field_type radius, const Dune::FieldVector<field_type, blocksize>& scaling) {

    radius_ = radius;

    for (size_t i=0; i<obstacles_.size(); i++) {

      for (int k=0; k<blocksize; k++) {

        obstacles_[i].lower(k) = -radius*scaling[k];
        obstacles_[i].upper(k) =  radius*scaling[k];

      }
    }

  }

  /** \brief Return true if the given vector is not contained in the trust region */
  bool violates(const Dune::BlockVector<Dune::FieldVector<double,blocksize> >& v) const
  {
    assert(v.size() == obstacles_.size());
    for (size_t i=0; i<v.size(); i++)
      for (size_t j=0; j<blocksize; j++)
        if (v[i][j] < obstacles_[i].lower(j) or v[i][j] > obstacles_[i].upper(j))
          return true;

    return false;
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

}  // namespace Dune::GFE

#endif
