#ifndef MULTI_INDEX_HH
#define MULTI_INDEX_HH

#include <vector>

/** \brief A multi-index
*/
class MultiIndex
    : public std::vector<unsigned int>
{

    // The range of each component
    unsigned int limit_;

public:
    /** \brief Constructor with a given range for each digit
     * \param n Number of digits
     */
    MultiIndex(unsigned int n, unsigned int limit)
        : std::vector<unsigned int>(n),
          limit_(limit)
    {
        std::fill(this->begin(), this->end(), 0);
    }

    /** \brief Increment the MultiIndex */
    MultiIndex& operator++() {

        for (size_t i=0; i<size(); i++) {

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
        for (size_t i=0; i<size(); i++)
            result *= limit_;
        return result;
    }

};

#endif