#ifndef UTOPIA_MODELS_OPDISC_AGING
#define UTOPIA_MODELS_OPDISC_AGING

#include "utils.hh"

namespace Utopia::Models::OpDisc::aging {

template<typename NWType, typename VertexDescType, typename RNGType>
void reinitialise_as_child( NWType& nw,
                            VertexDescType v,
                            bool extremism,
                            const double t,
                            RNGType& rng ){
    /** Reinitialises users as child vertices */
    auto parent = random_vertex(nw, rng);
    while (nw[parent].group<20 or nw[parent].group>40 or parent==v){
        parent = random_vertex(nw, rng);
    }
    nw[v].group = 10;
    nw[v].opinion = nw[parent].opinion;
    if (extremism) {
        nw[v].tolerance = utils::tolerance_func(nw[v].opinion, t);
    }
}

template<typename NWType, typename RNGType>
void user_revision( NWType& nw,
                    bool extremism,
                    const int life_expectancy,
                    const int peer_radius,
                    const double t,
                    RNGType& rng ){
    /** Chooses interaction partners, checks their groups and selects
      * the opinion update function */

    // choose random vertex pair that gets a revision opportunity
    auto v = random_vertex(nw, rng);
    auto nb = random_vertex(nw, rng);
    while (nb==v){ nb = random_vertex(nw, rng); }
    const double op_v = nw[v].opinion;
    const int age_difference = abs(nw[v].group-nw[nb].group);

    // the interaction between members of the same generation is always the same
    if (age_difference<peer_radius) {
        utils::update_opinion(v, nw[nb].opinion, nw);
        utils::update_opinion(nb, op_v, nw);
    }

    // directed conflict interaction: younger generations universally reject
    // older generations' opinions; older generations have a universally reduced
    // susceptibility towards younger generations' opinions.
    else {
        if (nw[v].group<nw[nb].group){
            utils::reject_opinion(v, nw[nb].opinion, nw);
            utils::update_opinion_disc(nb, op_v, nw);
        }
        else {
            utils::update_opinion_disc(v, nw[nb].opinion, nw);
            utils::reject_opinion(nb, op_v, nw);
        }
    }

    // update the tolerance
    if (extremism) {
        nw[v].tolerance = utils::tolerance_func(nw[v].opinion, t);
        nw[nb].tolerance = utils::tolerance_func(nw[nb].opinion, t);
    }

    // reinitialise users older than the life expectancy as children with
    // the opinion of a random parent (ages 20-40)
    if (nw[v].group>life_expectancy) {
        reinitialise_as_child(nw, v, extremism, t, rng);
    }
    else { ++nw[v].group; }

    if (nw[nb].group>life_expectancy) {
        reinitialise_as_child(nw, nb, extremism, t, rng);
    }
    else { ++nw[nb].group; }

} //user_revision

} // namespace

#endif // UTOPIA_MODELS_OPDISC_AGING
