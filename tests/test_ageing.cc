#define BOOST_TEST_MODULE test ageing

#include <boost/test/unit_test.hpp>
#include <utopia/core/model.hh>

#include "../aging.hh"
#include "../OpDisc.hh"

namespace Utopia::Models::OpDisc {

// ------------------------- Type definitions ----------------------------------
using Config = Utopia::DataIO::Config;
using vec_d = std::vector<double>;
std::mt19937 rng{};
std::uniform_real_distribution<double> uniform_prob_distr;

Config cfg = YAML::LoadFile("test_config.yml")["test_ageing"];

// ----------------------------- Fixtures --------------------------------------
struct TestNetwork {
    Network nw;
    TestNetwork() : nw{}
    {
        const unsigned num_vertices = 2;
        boost::generate_random_graph(nw, num_vertices, 0, rng, false, false);
    }
};

// ------------------------ Helper functions -----------------------------------
template<typename NWType>
void setup_nw (NWType& nw,
               const vec_d& groups,
               const vec_d& opinions,
               const vec_d& susc_1,
               const vec_d& tol,
               const vec_d& susc_2 = {0., 0.})
{
    /* Sets up the network with values from the lists passed
     * susceptibility_2 is 0 for all users by default
     */
    for (unsigned i=0; i<groups.size(); ++i) {
        nw[i].group = groups[i];
        nw[i].opinion = opinions[i];
        nw[i].susceptibility_1 = susc_1[i];
        nw[i].tolerance = tol[i];
        nw[i].susceptibility_2 = susc_2[i];
    }
}

template<typename NWType>
void test_group_ops (NWType& nw, int group, vec_d opinions) {
    /* Checks if the opinions of the group correspond to the given list
     * of opinions
     */
    for (auto v : range<IterateOver::vertices>(nw)) {
        if (nw[v].group==group) {
          BOOST_TEST (nw[v].opinion==opinions[v]);
        }
    }
}

double life_expectancy = 100;
double peer_radius = 10;
double time_scale = 1;

// ------------------------------ Tests ----------------------------------------
// test the age is incremented correctly
BOOST_FIXTURE_TEST_CASE (test_age_increase, TestNetwork) {
{
    vec_d groups = {10, 20};
    vec_d opinions = {0.1, 0.9};
    vec_d susc_1(2, 0.5);
    vec_d tol(2, 0.);
    int num_steps = 80;
    setup_nw(nw, groups, opinions, susc_1, tol);

    // let the model run so that no user reaches the life expectancy
    for (int i=0; i<num_steps; ++i) {
        aging::user_revision(nw, false, life_expectancy, peer_radius,
                             time_scale, 0., rng);
    }

    // check the ages have increased correctly
    for (auto v : range<IterateOver::vertices>(nw)) {
        BOOST_TEST (nw[v].group==groups[v]+num_steps);
    }
}
}

//------------------------------------------------------------------------------
// test the age is incremented correctly for various time scales
BOOST_FIXTURE_TEST_CASE (test_time_scales, TestNetwork,
                        * boost::unit_test::tolerance(1e-12)) {
{
    vec_d groups = {10, 20};
    vec_d opinions(groups.size(), 0.5);
    vec_d susc_1(groups.size(), 0.5);
    vec_d tol(groups.size(), 0.);

    //since we are operating on a small network, users cannot be reinitialised
    //as children. The number of steps need to be adjusted accordingly
    //to make sure user every crosses the life expectancy
    vec_d time_scales = get_as<vec_d>("time_scales", cfg);
    for (unsigned i=0; i<time_scales.size(); ++i) {

        setup_nw(nw, groups, opinions, susc_1, tol);

        //calculate max. possible number of steps
        int num_steps = ((life_expectancy-groups[1])/time_scales[i])-1;

        for (int j=0; j<num_steps; ++j) {
            aging::user_revision(nw, false, life_expectancy, peer_radius,
                                 time_scales[i], 0.5, rng);
        }

        for (int j=0; j<2; ++j) {
            BOOST_TEST (nw[j].group== groups[j]+(num_steps*time_scales[i]));
        }
    }
}
}

//------------------------------------------------------------------------------
// test users are correctly reinitialised as children
BOOST_FIXTURE_TEST_CASE (test_reinitialistion, TestNetwork) {
{
    vec_d groups = {15, 80};
    vec_d opinions = {0.5, 1};
    vec_d susc_1(2, 0);
    vec_d tol = {0.2, 0.4};
    int num_steps = 30;
    setup_nw(nw, groups, opinions, susc_1, tol);

    for (int i=0; i<num_steps; ++i) {
        aging::user_revision(nw, true, life_expectancy, peer_radius,
                             time_scale, 0.5, rng);
    }

    // check the older user has been reinitialised as a child with the
    // previously younger child as parent
    BOOST_TEST (nw[0].group==groups[0]+num_steps);
    BOOST_TEST (nw[1].group==18);
    BOOST_TEST (nw[0].opinion==opinions[0]);
    BOOST_TEST (nw[1].opinion==nw[0].opinion);
    BOOST_TEST (nw[0].tolerance==nw[1].tolerance);
}
}

//------------------------------------------------------------------------------
// test the opinion interaction process
BOOST_FIXTURE_TEST_CASE (test_interaction,
                         TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {
{
    vec_d groups = {10, 60};
    vec_d opinions = {0.5, 1};
    vec_d susc_1 = {0.5, 0.};
    vec_d tol = {1., 1.};
    vec_d susc_2 = {0., 0.75};
    int num_steps = 40;
    setup_nw(nw, groups, opinions, susc_1, tol, susc_2);

    // test younger users rejection older users, older users interacting
    // constructively with reduced suscepectibility
    for (int i=0; i<num_steps; ++i) {
        aging::user_revision(nw, false, life_expectancy, peer_radius,
                             time_scale, 0., rng);
    }
    BOOST_TEST (nw[0].opinion==nw[1].opinion);
    BOOST_TEST (nw[0].opinion<opinions[0]/2);

    // increase peer radius so that users are all part of one generation
    susc_1 = {0.5, 0.5};
    peer_radius = 51;
    setup_nw(nw, groups, opinions, susc_1, tol);

    for (int i=0; i<num_steps; ++i) {
        aging::user_revision(nw, false, life_expectancy, peer_radius,
                             time_scale, 0., rng);
    }

    //check only constructive interaction can take place
    BOOST_TEST (nw[0].opinion==nw[1].opinion);
    BOOST_TEST (nw[0].opinion==0.75);
}
}

} //namespace
