#ifndef UTOPIA_MODELS_OPDISC_HH
#define UTOPIA_MODELS_OPDISC_HH

#include <utopia/core/graph.hh>
#include <utopia/core/model.hh>
#include <utopia/data_io/graph_utils.hh>

#include "aging.hh"
#include "modes.hh"
#include "revision.hh"
#include "utils.hh"

namespace Utopia::Models::OpDisc {

using modes::Mode;
using modes::Mode::ageing;
using modes::Mode::conflict_dir;
using modes::Mode::conflict_undir;
using modes::Mode::isolated_1;
using modes::Mode::isolated_2;
using modes::Mode::reduced_int_prob;
using modes::Mode::reduced_s;

/*! Each user is a member of a group, may or may not discriminate in some wa
against members of other groups, holds and opinion, has a certain tolerance, and
is susceptible to other opinions. The discrimination may for some modes take the
form of reduced susceptibility to opinions from other groups (susceptibility_2)*/
struct User {
    double group;
    bool discriminates;
    double opinion;
    double tolerance;
    double susceptibility_1;  //same group interactions
    double susceptibility_2;  //inter-group interactions
};

/// The directed network type for the OpDisc Model:
using Network = boost::adjacency_list<
                boost::setS,        // edges
                boost::vecS,        // vertices
                boost::bidirectionalS,
                User>;               // vertex property

using OpDiscTypes = ModelTypes<>;

/// The OpDisc Model
template<Mode model_mode=reduced_int_prob>
class OpDisc:
    public Model<OpDisc<model_mode>, OpDiscTypes>
{
public:
    /// The base model type
    using Base = Model<OpDisc, OpDiscTypes>;

    /// Data type that holds the configuration
    using Config = typename Base::Config;

    /// Data type of the group to write model data to, holding datasets
    using DataGroup = typename Base::DataGroup;

    /// Data type for a dataset
    using DataSet = typename Base::DataSet;

    /// Data type of the shared RNG
    using RNG = typename Base::RNG;

private:
    // Base members: _time, _name, _cfg, _hdfgrp, _rng, _monitor

    std::uniform_real_distribution<double> _uniform_distr_prob_val;

    // User properties
    const Config _cfg_nw;
    Network _nw;
    const double _discriminators;
    const bool _extremism;
    const double _homophily_parameter;
    const double _life_expectancy;
    const unsigned _number_of_groups;
    const double _peer_radius;
    const double _susceptibility;
    const double _time_scale;
    const double _tolerance;

    // datasets and groups
    std::shared_ptr<DataGroup> _grp_nw;
    std::shared_ptr<DataSet> _dset_discriminators;
    std::shared_ptr<DataSet> _dset_group_label;
    std::shared_ptr<DataSet> _dset_opinion;
    std::shared_ptr<DataSet> _dset_users;

public:
    // Constructs the OpDisc model

    template<class ParentModel>
    OpDisc (const std::string name,
                    ParentModel &parent)
    :
        // Initialize first via base model
        Base(name, parent),
        _uniform_distr_prob_val(std::uniform_real_distribution<double>(0., 1.)),
        _cfg_nw(this->_cfg["nw"]),
        // initialize network
        _nw(this->init_nw()),
        // model parameters
        _discriminators(get_as<double>("discriminators", this->_cfg)),
        _extremism(get_as<bool>("extremism", this->_cfg)),
        _homophily_parameter(get_as<double>("homophily_parameter", this->_cfg)),
        _life_expectancy(get_as<double>("life_expectancy", this->_cfg["ageing"])),
        _number_of_groups(get_as<int>("number_of_groups", this->_cfg)),
        _peer_radius(get_as<double>("peer_radius", this->_cfg["ageing"])),
        _susceptibility(get_as<double>("susceptibility", this->_cfg)),
        _time_scale(get_as<double>("time_scale", this->_cfg["ageing"])),
        _tolerance(get_as<double>("tolerance", this->_cfg)),
        // create datagroups and datasets
        _grp_nw(Utopia::DataIO::create_graph_group(_nw, this->_hdfgrp, "nw")),
        _dset_discriminators(this->create_dset("discriminators", _grp_nw,
                                          {boost::num_vertices(_nw)}, 2)),
        _dset_group_label(this->create_dset("group_label", _grp_nw,
                                          {boost::num_vertices(_nw)}, 2)),
        _dset_opinion(this->create_dset("opinion", _grp_nw,
                                          {boost::num_vertices(_nw)}, 2)),
        _dset_users(this->create_dset("users", _grp_nw,
                                          {boost::num_vertices(_nw)}, 2))

    {
        this->_log->debug("Constructing the OpDisc Model ...");

        this->initialize_properties();

        this->_log->info("Initialized user network with {} vertices and {} edges",
                         num_vertices(_nw), num_edges(_nw));

        // Write the vertex data once as it does not change
        _dset_opinion->add_attribute("dim_name__1", "vertex");
        _dset_opinion->add_attribute("coords_mode__vertex", "trivial");

        _dset_discriminators->add_attribute("dim_name__1", "vertex");
        _dset_discriminators->add_attribute("coords_mode__vertex", "trivial");

        _dset_group_label->add_attribute("dim_name__1", "vertex");
        _dset_group_label->add_attribute("coords_mode__vertex", "trivial");
    }

public:

    // Setup functions .........................................................
    void initialize_properties() {
        this->_log->info("time scale: {}", _time_scale);
        this->_log->debug("Initializing user properties ...");
        utils::initialize<model_mode>(_nw,
                                      _discriminators,
                                      _extremism,
                                      _homophily_parameter,
                                      _life_expectancy,
                                      _number_of_groups,
                                      _susceptibility,
                                      _tolerance,
                                      _uniform_distr_prob_val,
                                      *this->_rng);
    } //initialize_properties
    Network init_nw() {
        this->_log->debug("Creating and initializing the user network ...");
        Network nw = Graph::create_graph<Network>(_cfg_nw, *this->_rng);
        return nw;
    }

public:
    // Runtime functions ......................................................
    void perform_step () {
        if constexpr (model_mode == ageing) {
             aging::user_revision (_nw,
                                  _extremism,
                                  _life_expectancy,
                                  _peer_radius,
                                  _time_scale,
                                  _tolerance,
                                  *this->_rng);
        }
        else {
            revision::user_revision<model_mode> (_nw,
                                                 _extremism,
                                                 _homophily_parameter,
                                                 _tolerance,
                                                 _uniform_distr_prob_val,
                                                 *this->_rng);
        }
    }

    void monitor () {}

    void write_data () {
        //Iterators
        auto [v, v_end] = boost::vertices(_nw);

        _dset_opinion->write(v, v_end,[this](auto vd) {
                                 return (float)_nw[vd].opinion;
                             });
        if constexpr (model_mode==ageing) {
          _dset_group_label->write(v, v_end, [this](auto vd) {
                                       return (float)_nw[vd].group;
                                   });
        }
        else if (this->get_time() + this->get_write_every() > this->get_time_max()) {
              _dset_discriminators->write(v, v_end, [this](auto vd) {
                                              return (unsigned) _nw[vd].discriminates;
                                          });
              _dset_group_label->write(v, v_end, [this](auto vd) {
                                           return (int) _nw[vd].group;
                                       });
              this->_log->debug("All datasets have been written!");
        }
    }
};

} //namespace

#endif // UTOPIA_MODELS_OPDISC_HH
