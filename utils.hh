#ifndef UTOPIA_MODELS_OPDISC_UTILS
#define UTOPIA_MODELS_OPDISC_UTILS

#include "modes.hh"

namespace Utopia::Models::OpDisc::utils{

using modes::Mode;
using modes::Mode::ageing;
using modes::Mode::conflict_dir;
using modes::Mode::conflict_undir;
using modes::Mode::isolated_1;
using modes::Mode::isolated_2;
using modes::Mode::reduced_int_prob;
using modes::Mode::reduced_s;

// RANDOM DISTRIBUTION UTILITY FUNCTIONS .......................................
template<typename RNGType>
int rand_int( int a, int b, RNGType& rng ) {
    /** Returns a random integer in [a, b] */
    std::uniform_int_distribution<int> distribution(a,b);
    return (int)distribution(rng);
}

template<typename RNGType>
double rand_double( double a, double b, RNGType& rng ) {
    /** Returns a random double in [a, b]*/
    std::uniform_real_distribution<double> distribution(a,b);
    return (double)distribution(rng);
}

template<typename RNGType>
double rand_double_Gaussian( double mu, double sigma, RNGType& rng ) {
    /** Returns a normally distributed double */
    std::normal_distribution<double> distribution(mu, sigma);
    return (double)distribution(rng);
}

// SETTERS .....................................................................
template<typename RNGType>
double set_init_Gauss( std::pair<double, double> distr_vals, RNGType& rng ) {
    /** Initialises a parameter with a normally distributed value in the interval [0, 1].
      * The distribution function is truncated at the edges.
      */
    double param = rand_double_Gaussian(distr_vals.first, distr_vals.second, rng);
    while (param<0 or param>1) {
       param = rand_double_Gaussian(distr_vals.first, distr_vals.second, rng);
    }
    return param;
}

template <Mode model_mode, typename RNGType>
double initialize_op( const int num_groups, const int group, RNGType& rng ) {
    /** Returns either a normally distributed value around the mean of the group,
      * or a random double in [0, 1]
      */
    if constexpr (model_mode==conflict_dir or model_mode==conflict_undir
                  or model_mode==ageing) {
        return rand_double(0, 1, rng);
    }
    else {
        double mean = group * 1./(num_groups-1);
        double stddev = 1./(2*(num_groups-1));
        std::pair<double, double> distr_values = std::make_pair(mean, stddev);
        return set_init_Gauss(distr_values, rng);
    }
}

double tolerance_func( const double opinion, const double tolerance_param) {
    /** Returns the tolerance as a function of the opinion. Users with extreme
    opinions will have a reduced tolerance. */
    return tolerance_param*(1-(2*pow((opinion-0.5),2)));
    //return tolerance_param*(2*pow((opinion-0.5), 2)+0.5);
}

// UPDATE FUNCTIONS ............................................................
double rejection_func( double op_1, double op_2, double susc ){
    /** Interaction function in the rejecting case
      * \return opinion The new opinion
      */
    if ((op_1<op_2) or (op_1==0. and op_2==0.)) {
        return op_1*(1.-susc*((op_2-op_1)/(1.-op_1)));
    }
    else {
        return op_1 + susc*((1.-op_1)*(op_1-op_2)/op_1);
    }
}

template <typename VertexDescType, typename NWType>
void reject_opinion( VertexDescType v, const double nb_op, NWType& nw ){
    /** The rejecting interaction. Users reject opinions to the
      * same degree they would otherwise agree with them
      */
    if (fabs(nw[v].opinion-nb_op)<=nw[v].tolerance) {
        nw[v].opinion = rejection_func(nw[v].opinion, nb_op, nw[v].susceptibility_1);
    }
}

template <typename VertexDescType, typename NWType>
void update_opinion( VertexDescType v, const double nb_op, NWType& nw ){
    /** Opinion update function without group dependency */
    if (fabs(nw[v].opinion-nb_op)<=nw[v].tolerance) {
        nw[v].opinion += nw[v].susceptibility_1 * (nb_op-nw[v].opinion);
    }
}

template <typename VertexDescType, typename NWType>
void update_opinion_disc( VertexDescType v, const double nb_op, NWType& nw ){
    /** Opinion update function with group dependency */
    if (fabs(nw[v].opinion-nb_op)<=nw[v].tolerance) {
        nw[v].opinion += nw[v].susceptibility_2 * (nb_op-nw[v].opinion);
    }
}

} // namespace

#endif // UTOPIA_MODELS_OPDISC_UTILS
