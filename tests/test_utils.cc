#define BOOST_TEST_MODULE test utils

#include <boost/test/unit_test.hpp>
#include <utopia/core/model.hh>

#include "../OpDisc.hh"
#include "../utils.hh"

namespace Utopia::Models::OpDisc {

// ------------------------- Type definitions ----------------------------------
using Config = Utopia::DataIO::Config;
using vec = std::vector<double>;
using vec_of_vec = std::vector<vec>;

std::mt19937 rng{};
std::uniform_real_distribution<double> uniform_prob_distr;
Config cfg = YAML::LoadFile("test_config.yml")["test_utils"];

// ----------------------------- Fixtures --------------------------------------
struct TestNetwork {
    using vertex = boost::graph_traits<Network>::vertex_descriptor;
    Network nw;
    TestNetwork()
    :
    nw{}
    {
        const unsigned num_vertices = get_as<int>("num_users", cfg["params"]);
        boost::generate_random_graph(nw, num_vertices, 0, rng, false, false);
        const double susc = get_as<double>("susceptibility", cfg["params"]);
        const double tol = get_as<double>("tolerance", cfg["params"]);
        const double p_hom = get_as<double>("homophily_parameter", cfg["params"]);
        for (auto v : range<IterateOver::vertices>(nw)) {
            nw[v].susceptibility_1 = susc;
            nw[v].susceptibility_2 = susc*p_hom;
            nw[v].tolerance = tol;
        }
    }
};

// ---------------------------- Helper function --------------------------------
template<typename NWType>
void assert_cases (Config& test_cfg,
                   NWType& nw,
                   void (*f)(TestNetwork::vertex, const double, NWType&))
{
    /* Runs the model with the function passed and checks the opinions against
    * the vaules from the cfg
    */
    //get opinion sets
    vec ops = get_as<vec>("opinions", test_cfg);
    vec nb_ops = get_as<vec>("nb_opinions", test_cfg);

    //get correct interaction results
    vec_of_vec to_assert = get_as<vec_of_vec>("to_assert", test_cfg);

    //get random interaction pair
    TestNetwork::vertex v = random_vertex(nw, rng);
    TestNetwork::vertex nb = random_vertex(nw, rng);
    while (nb==v) {
      nb = random_vertex(nw, rng);
    }

    //set opinion values, run interaction and check outcome
    for (unsigned i=0; i<ops.size(); ++i) {
        for (unsigned j=0; j<nb_ops.size(); ++j) {
            nw[v].opinion = ops[i];
            nw[nb].opinion = nb_ops[j];
            (*f)(v, nw[nb].opinion, nw);
            BOOST_TEST (nw[v].opinion==to_assert[j][i]);
        }
    }
}

// ---------------------------------- AUTO TESTS -------------------------------
// tests the tolerance update function, used for the 'extremism' mode
BOOST_AUTO_TEST_CASE (test_tolerance_func,
                      * boost::unit_test::tolerance(1e-12))
{
    Config test_cfg = cfg["test_funcs"]["test_tolerance_func"];
    double opinion = get_as<double>("opinion", test_cfg);

    vec tolerances = get_as<vec>("tolerances", test_cfg);
    vec to_assert = get_as<vec>("to_assert", test_cfg);

    for (unsigned i=0; i<tolerances.size(); ++i) {
        double tol = utils::tolerance_func(opinion, tolerances[i]);
        BOOST_TEST (tol==to_assert[i]);
    }
}

// -------------------------------- FIXTURE TESTS ------------------------------
// tests the opinion rejection function, used in the conflict and ageing modes
BOOST_FIXTURE_TEST_CASE (test_reject_op,
                         TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {
{
    Config test_cfg = cfg["test_funcs"]["test_reject_op"];
    assert_cases(test_cfg, nw, &utils::reject_opinion);
}
}

// tests the regular opinion update function
BOOST_FIXTURE_TEST_CASE (test_update_op,
                         TestNetwork,
                        * boost::unit_test::tolerance(1e-12)) {
{
    Config test_cfg = cfg["test_funcs"]["test_update_op"];
    assert_cases(test_cfg, nw, &utils::update_opinion);
}
}

// tests the discriminatory opinion update function
BOOST_FIXTURE_TEST_CASE (test_update_op_disc,
                         TestNetwork,
                         * boost::unit_test::tolerance(1e-12)){
{
    Config test_cfg = cfg["test_funcs"]["test_update_op_disc"];
    assert_cases(test_cfg, nw, &utils::update_opinion_disc);
}
}

} //namespace
