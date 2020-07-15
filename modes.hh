#ifndef UTOPIA_MODELS_OPDISC_MODES
#define UTOPIA_MODELS_OPDISC_MODES

namespace Utopia::Models::OpDisc::modes{

/** This class defines the various model types.
  * \tparam Mode model mode
  * \param ageing Directed conflict mode, with users ageing and slowly moving through age groups
  * \param conflict_dir Directed conflict mode: all groups reject each other's opinions
  * \param conflict_undir Undirected conflict mode: lower group numbers reject higher groups' opinions,
  * higher groups interact with lower group numbers with reduced susceptibility
  * \param isolated_1 Isolated discrimination (type 1): non-discriminators may interact with discriminators
  * \param isolated_2 Isolated discrimination (type 2): non-discriminators cannot interact with discriminators
  * \param reduced_int_prob Inter-group interactions take place with reduced probability
  * \param reduced_s Susceptibility is reduced for inter-group interactions
  */

enum Mode {
    ageing,
    conflict_dir,
    conflict_undir,
    isolated_1,
    isolated_2,
    reduced_int_prob,
    reduced_s
};

} //namespace

#endif // UTOPIA_MODELS_OPDISC_MODES
