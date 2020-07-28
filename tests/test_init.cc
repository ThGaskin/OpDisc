#define BOOST_TEST_MODULE test init

#include <boost/test/unit_test.hpp>
#include <utopia/core/model.hh>

#include "../modes.hh"
#include "../OpDisc.hh"
#include "../utils.hh"

namespace Utopia::Models::OpDisc {

// ------------------------- Type definitions ----------------------------------
using Config = Utopia::DataIO::Config;
using modes::Mode;

std::mt19937 rng{};
std::uniform_real_distribution<double> uniform_prob_distr;
Config config = YAML::LoadFile("test_config.yml")["test_init"];

// ----------------------------- Fixtures --------------------------------------
struct TestNetwork {
    Config cfg;
    Network nw;

    TestNetwork()
    :
    cfg(config),
    nw{}
    {
        const unsigned num_vertices = get_as<int>("num_users", cfg);
        boost::generate_random_graph(nw, num_vertices, 0, rng, false, false);
    }
};

// --------------------------- Helper function ---------------------------------
//Get the parameters from the cfg
const double discriminators = get_as<double>("discriminators", config);
const double homophily_parameter = get_as<double>("homophily_parameter", config);
const unsigned life_expectancy = get_as<unsigned>("life_expectancy", config);
const double susceptibility = get_as<double>("susceptibility", config);
const double tolerance = get_as<double>("tolerance", config);

std::vector<unsigned> num_groups
    = get_as<std::vector<unsigned>>("num_groups", config);

bool extremism = false;

template <Mode model_mode, typename NWType>
void initialize_model (NWType& nw, unsigned num_groups) {
    /* Initialises the model with the values from config */
    utils::initialize<model_mode>(nw,
                                  discriminators,
                                  extremism,
                                  homophily_parameter,
                                  life_expectancy,
                                  num_groups,
                                  susceptibility,
                                  tolerance,
                                  uniform_prob_distr,
                                  rng);
}

// ---------------------------- Tests ------------------------------------------

// -----------------------model_mode: ageing -----------------------------------
BOOST_FIXTURE_TEST_CASE (test_general_and_ageing, TestNetwork,
                         * boost::unit_test::tolerance(0.01)) {
{
    // loop over group numbers and check model initialisation
    for (unsigned n=0; n<num_groups.size(); ++n) {

        BOOST_TEST_CHECKPOINT("Test case: number of groups="<< num_groups[n]);

        initialize_model<ageing>(nw, num_groups[n]);

        //test opinion, susceptibility and tolerance initialisation
        //test discriminators initialisation by default
        BOOST_TEST_CHECKPOINT ("Testing general initialisation ...");
        double avg_op = 0;
        for (auto v : range<IterateOver::vertices>(nw)) {
            BOOST_TEST (nw[v].opinion>=0);
            BOOST_TEST (nw[v].opinion<=1);
            BOOST_TEST (nw[v].tolerance
                     == tolerance
                       );
            BOOST_TEST (nw[v].susceptibility_1
                     == susceptibility
                       );
            BOOST_TEST (nw[v].susceptibility_2
                     == susceptibility*homophily_parameter
                       );
            BOOST_TEST (nw[v].discriminates
                     == false
                       );
            avg_op+=nw[v].opinion;
        }
        avg_op/=boost::num_vertices(nw);
        BOOST_TEST (avg_op == 0.5);

        //test group initialisation in the ageing case
        BOOST_TEST_CHECKPOINT ("Testing ageing-specific properties ...");
        double avg_age = 0;
        for (auto v : range<IterateOver::vertices>(nw)) {
            BOOST_TEST (nw[v].group>=10);
            BOOST_TEST (nw[v].group<=life_expectancy);
            avg_age+=nw[v].group;
        }
        avg_age/=boost::num_vertices(nw);
        BOOST_TEST (avg_age == (life_expectancy+10)/2.);
    }
}
}

// -----------------------model_mode: conflict_dir -----------------------------
BOOST_FIXTURE_TEST_CASE (test_conflict_dir, TestNetwork,
                         * boost::unit_test::tolerance(0.01)) {
{
    // loop over group numbers and check model initialisation
    for (unsigned n=0; n<num_groups.size(); ++n) {

        initialize_model<conflict_dir>(nw, num_groups[n]);

        // test initialisation of susceptibility_2
        BOOST_TEST_CHECKPOINT ("Testing mode conflict_dir ...");
        std::vector<unsigned> groups(num_groups[n], 0);
        std::vector<double> group_op(num_groups[n], 0.);
        for (auto v : range<IterateOver::vertices>(nw)) {
            BOOST_TEST (nw[v].susceptibility_1
                     == susceptibility
                       );
            BOOST_TEST (nw[v].susceptibility_2
                     == susceptibility*homophily_parameter
                       );
            BOOST_TEST (nw[v].group>=0);
            BOOST_TEST (nw[v].group<=num_groups[n]-1);
            groups[nw[v].group]+=1;
            group_op[nw[v].group]+=nw[v].opinion;
        }

        // check all group sizes are equal and group opinions are
        // centered around 0.5
        for (unsigned i=0; i<groups.size(); ++i) {
            BOOST_TEST (1.*groups[i]
                     == boost::num_vertices(nw)/num_groups[n],
                        boost::test_tools::tolerance(0.025)
                       );
            double avg_op = group_op[i]/groups[i];
            BOOST_TEST (avg_op
                     == 0.5,
                        boost::test_tools::tolerance(0.02)
                       );
        }
    }
}
}

// -----------------------model_mode: conflict_undir ---------------------------
BOOST_FIXTURE_TEST_CASE (test_conflict_undir, TestNetwork,
                         * boost::unit_test::tolerance(0.01)) {
{
    // loop over group numbers and check model initialisation
    for (unsigned n=0; n<num_groups.size(); ++n) {

        initialize_model<conflict_undir>(nw, num_groups[n]);

        BOOST_TEST_CHECKPOINT ("Testing mode conflict_undir ...");

        // check proportion of discriminators
        double discriminator_prop = 0;
        for (auto v : range<IterateOver::vertices>(nw)){
            discriminator_prop+=nw[v].discriminates;
        }
        BOOST_TEST (discriminator_prop/boost::num_vertices(nw)
                 == discriminators
                   );
      }
}
}

// -----------------------model_mode: reduced_s --------------------------------
BOOST_FIXTURE_TEST_CASE (test_reduced_s, TestNetwork,
                         * boost::unit_test::tolerance(0.01)) {
{
    // loop over group numbers and check model initialisation
    for (unsigned n=0; n<num_groups.size(); ++n) {

        initialize_model<reduced_s>(nw, num_groups[n]);

        BOOST_TEST_CHECKPOINT ("Testing mode reduced_s ...");

        //collect group size and average opinion
        std::vector<unsigned> groups(num_groups[n], 0);
        std::vector<double> group_op(num_groups[n], 0.);
        for (auto v : range<IterateOver::vertices>(nw)) {
            BOOST_TEST (nw[v].group>=0);
            BOOST_TEST (nw[v].group<=num_groups[n]-1);
            groups[nw[v].group]+=1;
            group_op[nw[v].group]+=nw[v].opinion;
        }

        //check groups are evenly distributed and check group sizes
        for (unsigned i=1; i<groups.size()-1; ++i) {
            BOOST_TEST (1.*groups[i]
                     == boost::num_vertices(nw)/(num_groups[n]-1),
                        boost::test_tools::tolerance(0.04)
                       );
            double avg_op = group_op[i]/groups[i];
            if (num_groups[n]==1) {
                BOOST_TEST (avg_op
                         == 0.5,
                            boost::test_tools::tolerance(0.02)
                           );
            }
            else {
              BOOST_TEST (avg_op
                       == 1.*i/(num_groups[n]-1),
                          boost::test_tools::tolerance(0.04)
                         );
            }

        }
        BOOST_TEST (group_op[0]/groups[0]
                 == 1.-group_op.back()/groups.back(),
                    boost::test_tools::tolerance(0.04)
                   );
        BOOST_TEST (1.*groups[0]
                 == 1.*groups.back(),
                    boost::test_tools::tolerance(0.04)
                   );
    }
}
}

// ----------------------- extremism: true -------------------------------------
BOOST_FIXTURE_TEST_CASE (test_extremism, TestNetwork) {
{
    // loop over group numbers and check model initialisation
    extremism = true;

    initialize_model<reduced_s>(nw, 2);

    BOOST_TEST_CHECKPOINT ("Testing model with extremism on ...");

    for (auto v : range<IterateOver::vertices>(nw)) {
            BOOST_TEST (nw[v].tolerance
                     == utils::tolerance_func(nw[v].opinion, tolerance));
        }
}
}

} //namespace
