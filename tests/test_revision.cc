#define BOOST_TEST_MODULE test revision

#include <boost/test/unit_test.hpp>

#include <utopia/core/model.hh>

#include "../OpDisc.hh"
#include "../revision.hh"
#include "../utils.hh"

namespace Utopia::Models::OpDisc {

// --------------------------- Type definitions --------------------------------
using vec_b = std::vector<bool>;
using vec_d = std::vector<double>;
using vec_u = std::vector<unsigned>;
std::mt19937 rng{};
std::uniform_real_distribution<double> uniform_prob_distr;


// ------------------------------ Fixtures -------------------------------------
struct Large_TestNetwork {
    Network nw;
    Large_TestNetwork() : nw{}
    {
        const unsigned num_vertices = 4;
        boost::generate_random_graph(nw, num_vertices, 0, rng, false, false);
    }
};

struct Small_TestNetwork {
    Network nw;
    Small_TestNetwork() : nw{}
    {
        const unsigned num_vertices = 2;
        boost::generate_random_graph(nw, num_vertices, 0, rng, false, false);
    }
};

// ------------------------- Helper functions ----------------------------------
template<typename NWType>
void setup_nw (NWType& nw,
               const vec_u& groups,
               const vec_d& opinions,
               const vec_d& susc_1,
               const vec_d& tol,
               const vec_b& discriminates = {false, false, false, false},
               const vec_d& susc_2 = {0., 0., 0., 0.})
{
    for (unsigned i=0; i<groups.size(); ++i) {
        nw[i].group = groups[i];
        nw[i].opinion = opinions[i];
        nw[i].susceptibility_1 = susc_1[i];
        nw[i].tolerance = tol[i];
        nw[i].discriminates = discriminates[i];
        nw[i].susceptibility_2 = susc_2[i];
    }
}

template<typename NWType>
void test_group_ops (NWType& nw, int group, vec_d opinions) {
    for (auto v : range<IterateOver::vertices>(nw)) {
        if (nw[v].group==group) {
            BOOST_TEST (nw[v].opinion==opinions[v]);
        }
    }
}

template<typename NWType>
double op_sum (NWType& nw, const unsigned num_users) {
    double sum = 0;
    for (unsigned i=0; i<num_users; ++i) {
        sum+=nw[i].opinion;
    }
    return sum;
}

// ---------------------------- Tests ------------------------------------------
//test the opinion update function of the reduced interaction probability mode
BOOST_FIXTURE_TEST_CASE (test_reduced_int_prob_op_update,
                         Large_TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {
{
    vec_u groups = {0, 0, 1, 1};
    std::vector<vec_d> opinions = {{0., 1., 0.256, 0.453},
                                   {0.1, 0.9, 0.4, 0.6},
                                   {0.8, 0.4, 1, 0.7}};
    vec_d susc_1 = {0.25, 0.25, 1., 1.};
    vec_d tol = {1., 1., 0.3, 0.3};
    double p_hom = 1.;

    //possible opinions after one interaction
    std::vector<vec_d> ops_after_one_int = {{0.25, 0.75, 0.453, 0.256},
                                            {0.3, 0.6, 0.6, 0.4},
                                            {0.7, 0.5, 0.7, 1}};
    //possible opinions after two interactions
    std::vector<vec_d> ops_after_two_int = {{0.375, 0.625, 0.256, 0.453},
                                            {0.375, 0.525, 0.4, 0.6},
                                            {0.65, 0.55, 1, 0.7}};

    for (unsigned i=0; i<opinions.size(); ++i) {

        BOOST_TEST_CHECKPOINT ("Test 1." << i);

        setup_nw(nw, groups, opinions[i], susc_1, tol);
        std::pair<int, int> interaction_combo = std::make_pair(1, 1);

        revision::user_revision<reduced_int_prob>(nw, false, p_hom, 0.,
                                                  uniform_prob_distr, rng);

        //first group interacted in the first step
        if (nw[0].opinion!=opinions[i][0]) { interaction_combo.first = 0; }

        revision::user_revision<reduced_int_prob>(nw, false, p_hom, 0.,
                                                  uniform_prob_distr, rng);

        //second group interacted in the second step
        if (nw[0].opinion==ops_after_two_int[i][0] or
           (nw[0].opinion==ops_after_one_int[i][0]
             and interaction_combo.first==1)) {
            interaction_combo.second = 0;
        }

        switch (abs(interaction_combo.first-interaction_combo.second)) {
            case(0): {
                //same group interacted both times
                BOOST_TEST_CHECKPOINT ("Same group interacted twice");
                test_group_ops(nw, interaction_combo.second, ops_after_two_int[i]);
                test_group_ops(nw, abs(interaction_combo.second-1), opinions[i]);
                break;
            }
            case(1): {
                BOOST_TEST_CHECKPOINT ("Both groups interacted once");
                //both groups interacted once
                test_group_ops(nw, interaction_combo.first, ops_after_one_int[i]);
                test_group_ops(nw, interaction_combo.second, ops_after_one_int[i]);
                break;
            }
        }
    }

    BOOST_TEST_CHECKPOINT ("Test 2");

    groups = {0, 1, 2, 3};
    vec_d ops = {0., 0.33, 0.66, 1.};
    susc_1 = {0.1, 0.2, 0.3, 0.4};
    tol = {1., 1., 1., 1.};
    p_hom = 0;
    setup_nw(nw, groups, ops, susc_1, tol);

    BOOST_TEST (op_sum(nw, boost::num_vertices(nw))==1.99);

    revision::user_revision<reduced_int_prob>(nw, false, p_hom, 0,
                                              uniform_prob_distr, rng);

    //check an interaction took place
    BOOST_TEST (op_sum(nw, boost::num_vertices(nw))!=1.99);

    const double& opsum = op_sum(nw, boost::num_vertices(nw));
    revision::user_revision<reduced_int_prob>(nw, false, p_hom, 0,
                                              uniform_prob_distr, rng);
    //check an interaction took place
    BOOST_TEST (opsum!=op_sum(nw, boost::num_vertices(nw)));
}
}

// -----------------------------------------------------------------------------
// test the opinion update function of the isolated_1 mode
BOOST_FIXTURE_TEST_CASE (test_isolated1_op_update,
                         Large_TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {
{
    vec_u groups = {0, 0, 1, 1};
    vec_d opinions = {0., 0., 0.5, 0.5};
    vec_d susc_1(4, 1.);
    vec_d tol(4, 1.);
    vec_b discriminates = {true, true, false, false};
    setup_nw(nw, groups, opinions, susc_1, tol, discriminates);

    for (unsigned i=0; i<4; ++i) {
        revision::user_revision<isolated_1>(nw, false, 0, 2,
                                            uniform_prob_distr, rng);
    }

    //check group 2 interacted with group 1
    test_group_ops(nw, 0, opinions);
    for (unsigned i=2; i<4; ++i){
       if (nw[i].opinion!=opinions[i]) {
           BOOST_TEST (nw[i].opinion==0.);
       }
    }

    groups = {0, 1, 2, 3};
    opinions = {0.2, 0.2, 0.2, 0.75};
    discriminates = {true, true, true, false};
    setup_nw(nw, groups, opinions, susc_1, tol, discriminates);

    //check only group 4 can interact
    for (unsigned i=0; i<4; ++i) {
        revision::user_revision<isolated_1>(nw, false, 0, 1,
                                            uniform_prob_distr, rng);
        if (i!=3) {
          test_group_ops(nw, i, opinions);
        }
        else {
            if (nw[3].opinion!=opinions[3]) {
                BOOST_TEST (nw[3].opinion==opinions[0]);
            }
        }
    }
}
}

// -----------------------------------------------------------------------------
// test the opinion update function of the isolated_2 mode
BOOST_FIXTURE_TEST_CASE (test_isolated2_op_update,
                         Large_TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {
{
    vec_u groups = {0, 0, 1, 1};
    vec_d opinions = {0., 0.5, 0.5, 1.};
    vec_d susc_1(4, 0.5);
    vec_d tol(4, 1.);
    vec_b discriminates = {true, true, false, false};
    setup_nw(nw, groups, opinions, susc_1, tol, discriminates);

    for (unsigned i=0; i<4; ++i) {
        revision::user_revision<isolated_2>(nw, false, 0, 1,
                                            uniform_prob_distr, rng);
    }

    vec_d ops_after_one_int = {0.25, 0.25, 0.75, 0.75};
    if (nw[0].opinion!=opinions[0]) {
        test_group_ops(nw, 0, ops_after_one_int);
    }

    if (nw[2].opinion!=opinions[2]) {
      test_group_ops(nw, 1, ops_after_one_int);
    }
}
}

// -----------------------------------------------------------------------------
// test the opinion update function of the reduced_s mode
BOOST_FIXTURE_TEST_CASE (test_reduced_s_op_update,
                         Large_TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {
{
    vec_u groups = {0, 0, 1, 1};
    vec_d opinions = {0., 0., 1., 1.};
    vec_d susc_1(4, 0.);
    vec_d tol(4, 1.);
    vec_b discriminates(4, true);
    std::vector<vec_d> susc_2 = {{0.5, 0.5, 0.5, 0.5},
                                 {0.25, 0.25, 0.25, 0.25},
                                 {0.75, 0.75, 0.75, 0.75}};

    //opinion values after interactions
    std::vector<vec_d> ops_after_ints = {{0.5, 0.5, 0.5, 0.5},
                                         {0.25, 0.25, 0.75, 0.75},
                                         {0.75, 0.75, 0.25, 0.25}};

    //check interactions took place using the correct susceptibilities
    for (unsigned i=0; i<3; ++i){
        setup_nw(nw, groups, opinions, susc_1, tol, discriminates, susc_2[i]);
        revision::user_revision<reduced_s>(nw, false, 0., 1,
                                           uniform_prob_distr, rng);
        for (unsigned j=0; j<4; ++j){
            if (nw[j].opinion!=opinions[j]){
                BOOST_TEST (nw[j].opinion==ops_after_ints[i][j]);
            }
        }
    }
}
}

// -----------------------------------------------------------------------------
// test the opinion update function of the conflict_dir mode
BOOST_FIXTURE_TEST_CASE (test_conflict_dir,
                         Small_TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {
{
    vec_u groups = {0, 1};
    std::vector<vec_d> opinions = {{0.3, 0.7}, {0.2, 0.5}, {0.2, 0.4},
                                   {0.5, 1.}, {0.75, 0.75}, {0., 1.},
                                   {0.5, 0.}, {0.75, 0.25}};
    vec_d tol(2, 1.);
    vec_d susc_1 = {0.5, 0.};
    vec_b discriminates(2, true);
    vec_d susc_2 = {0., 0.5};

    //the opinion values after each interaction
    std::vector<vec_d> ops_after_ints = {{1.5/7, 0.5}, {0.1625, 0.35},
                                         {0.175, 0.3}, {0.25, 0.75},
                                         {0.75, 0.75}, {0., 0.5}, {0.75, 0.25},
                                         {5./6, 0.5}};

    //set up network with opinions, allow users to interact once, then check
    //the resulting opinions match the prediction from ops_after_ints
    for (unsigned i=0; i<8; ++i) {
         setup_nw(nw, groups, opinions[i], susc_1, tol, discriminates, susc_2);
         revision::user_revision<conflict_dir>(nw, false, 1, 1,
                                               uniform_prob_distr, rng);
         for (unsigned j=0; j<groups.size(); ++j) {
              test_group_ops(nw, j, ops_after_ints[i]);
         }
    }
}
}

// -----------------------------------------------------------------------------
// test the opinion update function of the conflict_undir mode
BOOST_FIXTURE_TEST_CASE (test_conflict_undir,
                         Small_TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {
{
    vec_u groups = {0, 1};
    std::vector<vec_d> opinions = {{0.4, 0.7}, {0., 0.6}, {0., 0.8},
                                   {0., 0.9}, {0., 1.}, {0., 0.}};
    vec_d susc_1(2, 0.5);
    vec_d tol(2, 1.);
    vec_b discriminates(2, true);

    //opinions after one interaction
    std::vector<vec_d> ops_after_ints = {{0.3, 0.7642857142857142}, {0., 0.8},
                                         {0., 0.9}, {0., 0.95}, {0., 1.},
                                         {0., 0.}};

    //set up network with opinions, allow users to interact once, then check
    //the resulting opinions match the prediction from ops_after_ints
    for (unsigned i=0; i<6; ++i){
         setup_nw(nw, groups, opinions[i], susc_1, tol, discriminates);
         revision::user_revision<conflict_undir>(nw, false, 1, 1,
                                                 uniform_prob_distr, rng);
         for (unsigned j=0; j<groups.size(); ++j){
              test_group_ops(nw, j, ops_after_ints[i]);
         }
    }
}
}

// -----------------------------------------------------------------------------
// test the opinion update function of the reduced_s mode with extremism on
BOOST_FIXTURE_TEST_CASE (test_extremism,
                         Small_TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {
{
    vec_u groups = {0, 0};
    vec_d opinions = {0.3, 0.6};
    vec_d susc_1(2, 0.2);
    vec_d tol(2, 0.3);

    setup_nw(nw, groups, opinions, susc_1, tol);
    revision::user_revision<reduced_s>(nw, true, 0., tol[0],
                                       uniform_prob_distr, rng);

    BOOST_TEST (nw[0].tolerance = 0.28824);
    BOOST_TEST (nw[1].tolerance = 0.28824);
}
}

} // namespace
