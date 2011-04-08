#ifndef MULTI_INDEX_HH
#define MULTI_INDEX_HH

#include <dune/common/array.hh>

/** \brief N-dimensional multi-index
*/
template <int N>
class MultiIndex
    : public Dune::array<unsigned int,N>
{

    // The range of each component
    unsigned int limit_;

public:
    /** \brief Constructor with a given range for each digit */
    MultiIndex(unsigned int limit)
        : limit_(limit)
    {
        std::fill(this->begin(), this->end(), 0);
    }

    /** \brief Increment the MultiIndex */
    MultiIndex& operator++() {

        for (int i=0; i<N; i++) {

            // Augment digit
            (*this)[i]++;

            // If there is no carry-over we can stop here
            if ((*this)[i]<limit_)
                break;

            (*this)[i] = 0;
                    
        }
        return *this;
    }

    /** \brief Compute how many times you can call operator++ before getting to (0,...,0) again */
    size_t cycle() const {
        size_t result = 1;
        for (int i=0; i<N; i++)
            result *= limit_;
        return result;
    }

};

#endif