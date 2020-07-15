#ifndef UTOPIA_MODELS_OPDISC_REVISION
#define UTOPIA_MODELS_OPDISC_REVISION

#include "modes.hh"
#include "utils.hh"

namespace Utopia::Models::OpDisc::revision {

using modes::Mode;
using modes::Mode::conflict_dir;
using modes::Mode::conflict_undir;
using modes::Mode::isolated_1;
using modes::Mode::isolated_2;
using modes::Mode::reduced_int_prob;
using modes::Mode::reduced_s;

template<Mode model_mode, typename NWType, typename RNGType>
void user_revision( NWType& nw,
                    const bool extremism,
                    const double homophily_param,
                    const double t,
                    std::uniform_real_distribution<double> prob_distr,
                    RNGType& rng ){
    /** Checks the model mode, chooses interaction partners and selects
      * the opinion update function */

    // choose random vertex pair that gets a revision opportunity
    auto v = random_vertex(nw, rng);
    auto nb = random_vertex(nw, rng);
    while (nb==v){ nb = random_vertex(nw, rng); }
    const double op_v = nw[v].opinion;

    // The interaction between members of the same group is always the same
    if (nw[v].group==nw[nb].group) {
        utils::update_opinion(v, nw[nb].opinion, nw);
        utils::update_opinion(nb, op_v, nw);
    }

    // Directed conflict interaction: lower group numbers universally reject higher groups'
    // opinions, higher group numbers universally discriminate against lower groups' opinions
    if constexpr (model_mode==conflict_dir) {
        if (nw[v].group<nw[nb].group) {
            utils::reject_opinion(v, nw[nb].opinion, nw);
            utils::update_opinion_disc(nb, op_v, nw);
        }
        else {
            utils::update_opinion_disc(v, nw[nb].opinion, nw);
            utils::reject_opinion(nb, op_v, nw);
        }
    }

    else if constexpr (model_mode==conflict_undir) {
        if (nw[v].discriminates){
            utils::reject_opinion(v, nw[nb].opinion, nw);
        }
        else{
            utils::update_opinion_disc(v, nw[nb].opinion, nw);
        }
        if (nw[nb].discriminates) {
            utils::reject_opinion(nb, op_v, nw);
        }
        else{
            utils::update_opinion_disc(nb, op_v, nw);
        }
    }

    else if constexpr (model_mode==isolated_1) {
        if (not nw[v].discriminates) {
            utils::update_opinion(v, nw[nb].opinion, nw);
        }
        if (not nw[nb].discriminates) {
            utils::update_opinion(nb, op_v, nw);
        }
    }

    else if constexpr (model_mode==isolated_2) {
        if (not nw[v].discriminates and not nw[nb].discriminates) {
            utils::update_opinion(v, nw[nb].opinion, nw);
            utils::update_opinion(nb, op_v, nw);
        }
    }

    else if constexpr (model_mode==reduced_int_prob) {
        const double interaction_prob=prob_distr(rng);
        if (interaction_prob<=homophily_param){
            while(nw[v].group!=nw[nb].group or nb==v) {
                nb = random_vertex(nw, rng);
            }
        }
        utils::update_opinion(v, nw[nb].opinion, nw);
        utils::update_opinion(nb, op_v, nw);
    }

    else if constexpr (model_mode==reduced_s) {
        utils::update_opinion_disc(v, nw[nb].opinion, nw);
        utils::update_opinion_disc(nb, op_v, nw);
    }
    if (extremism) {
       nw[v].tolerance = utils::tolerance_func(nw[v].opinion, t);
       nw[nb].tolerance = utils::tolerance_func(nw[nb].opinion, t);
    }
}

} // namespace

#endif // UTOPIA_MODELS_OPDISC_REVISION
