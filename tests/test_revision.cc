#define BOOST_TEST_MODULE test revision

#include <boost/test/unit_test.hpp>

#include <utopia/core/model.hh>

#include "../OpDisc.hh"
#include "../revision.hh"
#include "../utils.hh"


namespace Utopia::Models::OpDisc {

using modes::Mode;
using modes::Mode::conflict_dir;
using modes::Mode::conflict_undir;
using modes::Mode::isolated_1;
using modes::Mode::isolated_2;
using modes::Mode::reduced_int_prob;
using modes::Mode::reduced_s;

// -- Type definitions --------------------------------------------------------

std::mt19937 rng{};
std::uniform_real_distribution<double> uniform_prob_distr;


// -- Fixtures ----------------------------------------------------------------

// Test random network
struct TestNetwork {
    using vertex = boost::graph_traits<Network>::vertex_descriptor;
    using Config = Utopia::DataIO::Config;

    Config cfg;
    Network nw;

    TestNetwork()
    :
    cfg(YAML::LoadFile("test_config.yml")),
    nw{}
    {
        const unsigned num_edges = 0;
        const int num_vertices=4;
        constexpr bool allow_parallel = true;
        constexpr bool allow_self_edges = false;

        boost::generate_random_graph(nw,
                                 num_vertices,
                                 num_edges,
                                 rng,
                                 allow_parallel,
                                 allow_self_edges);
    }

};

struct TestNetwork2 {
    using vertex = boost::graph_traits<Network>::vertex_descriptor;
    using Config = Utopia::DataIO::Config;

    Config cfg;
    Network nw;

    TestNetwork2()
    :
    cfg(YAML::LoadFile("test_config.yml")),
    nw{}
    {
        const unsigned num_edges = 0;
        const int num_vertices=2;
        constexpr bool allow_parallel = true;
        constexpr bool allow_self_edges = false;

        boost::generate_random_graph(nw,
                                 num_vertices,
                                 num_edges,
                                 rng,
                                 allow_parallel,
                                 allow_self_edges);
    }

};

// -- Actual test -------------------------------------------------------------


//2. Test opinion update........................................................
BOOST_FIXTURE_TEST_CASE( test_reduced_int_prob_op_update,
                         TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {

    {
        vertex v = 0;
        vertex w = 1;
        vertex x = 2;
        vertex y = 3;

        nw[v].discriminates = false;
        nw[w].discriminates = false;
        nw[x].discriminates = false;
        nw[y].discriminates = false;

        nw[v].group = 0;
        nw[w].group = 0;
        nw[x].group = 1;
        nw[y].group = 1;

        nw[v].opinion = 0.;
        nw[w].opinion = 1.;
        nw[x].opinion = 0.3;
        nw[y].opinion = 0.6;

        nw[v].susceptibility_1 = 0.25;
        nw[w].susceptibility_1 = 0.25;
        nw[x].susceptibility_1 = 1;
        nw[y].susceptibility_1 = 1;

        nw[v].susceptibility_2 = 0;
        nw[w].susceptibility_2 = 0;
        nw[x].susceptibility_2 = 0;
        nw[y].susceptibility_2 = 0;

        nw[v].tolerance = 1.;
        nw[w].tolerance = 1.;
        nw[x].tolerance = .3;
        nw[y].tolerance = .3;

        revision::user_revision<reduced_int_prob>(nw, true, 1, 1, uniform_prob_distr, rng);

        bool group_0_int = false;
        if(nw[v].opinion!=0){
          BOOST_TEST(nw[v].opinion == .25);
          BOOST_TEST(nw[w].opinion == .75);
          group_0_int = true;
        }
        else{
          BOOST_TEST(nw[x].opinion == .6);
          BOOST_TEST(nw[y].opinion == .3);
        }

        revision::user_revision<reduced_int_prob>(nw, true, 1, 1, uniform_prob_distr, rng);

        if (group_0_int){
          if(nw[v].opinion!=0.25){
            BOOST_TEST(nw[v].opinion == .5);
            BOOST_TEST(nw[w].opinion == .5);
          }
          else{
            BOOST_TEST(nw[x].opinion == .3);
            BOOST_TEST(nw[y].opinion == .6);
          }
        }
        else{
          if(nw[v].opinion!=0.){
            BOOST_TEST(nw[v].opinion == .25);
            BOOST_TEST(nw[w].opinion == .75);
          }
          else{
            BOOST_TEST(nw[x].opinion == .3);
            BOOST_TEST(nw[y].opinion == .6);
          }
        }

        nw[v].group = 0;
        nw[w].group = 1;
        nw[x].group = 2;
        nw[y].group = 3;

        nw[v].opinion = 0.;
        nw[w].opinion = 0.33;
        nw[x].opinion = 0.66;
        nw[y].opinion = 1;

        nw[v].susceptibility_1 = .1;
        nw[w].susceptibility_1 = .2;
        nw[x].susceptibility_1 = .3;
        nw[y].susceptibility_1 = .4;

        nw[v].susceptibility_2 = 0;
        nw[w].susceptibility_2 = 0;
        nw[x].susceptibility_2 = 0;
        nw[y].susceptibility_2 = 0;

        nw[v].tolerance = 1.;
        nw[w].tolerance = 1.;
        nw[x].tolerance = 1.;
        nw[y].tolerance = 1.;

        double op_sum = nw[v].opinion+nw[w].opinion+nw[x].opinion+nw[y].opinion;
        BOOST_TEST(op_sum==1.99);
        revision::user_revision<reduced_int_prob>(nw, true, 0, 2, uniform_prob_distr, rng);

        op_sum = nw[v].opinion+nw[w].opinion+nw[x].opinion+nw[y].opinion;
        BOOST_TEST(op_sum != 1.99);
        op_sum = nw[v].opinion+nw[w].opinion+nw[x].opinion+nw[y].opinion;
        revision::user_revision<reduced_int_prob>(nw, true, 0, 2, uniform_prob_distr, rng);

        double op_sum_2 = nw[v].opinion+nw[w].opinion+nw[x].opinion+nw[y].opinion;
        BOOST_TEST(op_sum_2 != op_sum);
    }
}

BOOST_FIXTURE_TEST_CASE( test_isolated1_op_update,
                         TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {

    {
      vertex v = 0;
      vertex w = 1;
      vertex x = 2;
      vertex y = 3;

      nw[v].group = 0;
      nw[w].group = 0;
      nw[x].group = 1;
      nw[y].group = 1;

      nw[v].opinion = 0.;
      nw[w].opinion = 0.;
      nw[x].opinion = 0.5;
      nw[y].opinion = 0.5;

      nw[v].susceptibility_1 = 1;
      nw[w].susceptibility_1 = 1;
      nw[x].susceptibility_1 = 1;
      nw[y].susceptibility_1 = 1;

      nw[v].susceptibility_2 = 0;
      nw[w].susceptibility_2 = 0;
      nw[x].susceptibility_2 = 0;
      nw[y].susceptibility_2 = 0;

      nw[v].tolerance = 1;
      nw[w].tolerance = 1;
      nw[x].tolerance = 1;
      nw[y].tolerance = 1;

      nw[v].discriminates = true;
      nw[w].discriminates = true;
      nw[x].discriminates = false;
      nw[y].discriminates = false;

      revision::user_revision<isolated_1>(nw, true, 0, 2, uniform_prob_distr, rng);
      revision::user_revision<isolated_1>(nw, true, 0, 2, uniform_prob_distr, rng);
      revision::user_revision<isolated_1>(nw, true, 0, 2, uniform_prob_distr, rng);

      BOOST_TEST(nw[v].opinion==0.);
      BOOST_TEST(nw[w].opinion==0.);
      if (nw[x].opinion!=0.5){
        BOOST_TEST(nw[x].opinion==0.);
      }
      if (nw[y].opinion!=0.5){
        BOOST_TEST(nw[y].opinion==0.);
      }

      nw[v].group = 0;
      nw[w].group = 1;
      nw[x].group = 2;
      nw[y].group = 3;

      nw[v].opinion = 0.2;
      nw[w].opinion = 0.2;
      nw[x].opinion = 0.2;
      nw[y].opinion = 0.75;

      nw[v].discriminates = true;
      nw[w].discriminates = true;
      nw[x].discriminates = true;
      nw[y].discriminates = false;

      revision::user_revision<isolated_1>(nw, true, 0, 2, uniform_prob_distr, rng);
      revision::user_revision<isolated_1>(nw, true, 0, 2, uniform_prob_distr, rng);
      revision::user_revision<isolated_1>(nw, true, 0, 2, uniform_prob_distr, rng);
      revision::user_revision<isolated_1>(nw, true, 0, 2, uniform_prob_distr, rng);

      BOOST_TEST(nw[v].opinion==0.2);
      BOOST_TEST(nw[w].opinion==0.2);
      BOOST_TEST(nw[x].opinion==0.2);
      if(nw[y].opinion!=0.75){
        BOOST_TEST(nw[y].opinion==0.2);
      }
    }
}

BOOST_FIXTURE_TEST_CASE( test_isolated2_op_update,
                         TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {

    {
      vertex v = 0;
      vertex w = 1;
      vertex x = 2;
      vertex y = 3;

      nw[v].group=0;
      nw[w].group=0;
      nw[x].group=1;
      nw[y].group=1;

      nw[v].discriminates = true;
      nw[w].discriminates = true;
      nw[x].discriminates = false;
      nw[y].discriminates = false;

      nw[v].opinion =0;
      nw[w].opinion=0.5;
      nw[x].opinion=0.5;
      nw[y].opinion=1.;

      nw[v].tolerance=1;
      nw[w].tolerance=1;
      nw[x].tolerance=1;
      nw[y].tolerance=1;

      nw[v].susceptibility_1=.5;
      nw[w].susceptibility_1=.5;
      nw[x].susceptibility_1=.5;
      nw[y].susceptibility_1=.5;

      nw[v].susceptibility_2 = 0;
      nw[w].susceptibility_2 = 0;
      nw[x].susceptibility_2 = 0;
      nw[y].susceptibility_2 = 0;

      revision::user_revision<isolated_2>(nw, true, 0, 2, uniform_prob_distr, rng);
      revision::user_revision<isolated_2>(nw, true, 0, 2, uniform_prob_distr, rng);
      revision::user_revision<isolated_2>(nw, true, 0, 2, uniform_prob_distr, rng);
      revision::user_revision<isolated_2>(nw, true, 0, 2, uniform_prob_distr, rng);

      if(nw[v].opinion!=0.){
        BOOST_TEST(nw[v].opinion==0.25);
        BOOST_TEST(nw[w].opinion==0.25);
      }
      if(nw[x].opinion!=0.5){
        BOOST_TEST(nw[x].opinion==0.75);
        BOOST_TEST(nw[y].opinion==0.75);
      }

    }
}

BOOST_FIXTURE_TEST_CASE( test_reduced_s_op_update,
                         TestNetwork,
                         * boost::unit_test::tolerance(1e-12)) {

    {
      vertex v = 0;
      vertex w = 1;
      vertex x = 2;
      vertex y = 3;

      nw[v].group=0;
      nw[w].group=0;
      nw[x].group=1;
      nw[y].group=1;

      nw[v].discriminates = true;
      nw[w].discriminates = true;
      nw[x].discriminates = true;
      nw[y].discriminates = true;

      nw[v].opinion=0;
      nw[w].opinion=0.;
      nw[x].opinion=1.;
      nw[y].opinion=1.;

      nw[v].tolerance=1;
      nw[w].tolerance=1;
      nw[x].tolerance=1;
      nw[y].tolerance=1;

      nw[v].susceptibility_1 = 0;
      nw[w].susceptibility_1 = 0;
      nw[x].susceptibility_1 = 0;
      nw[y].susceptibility_1 = 0;

      nw[v].susceptibility_2=0.5;
      nw[w].susceptibility_2=0.5;
      nw[x].susceptibility_2=0.5;
      nw[y].susceptibility_2=0.5;

      revision::user_revision<reduced_s>(nw, true, 0.5, 2, uniform_prob_distr, rng);

      if(nw[v].opinion!=0){
        BOOST_TEST(nw[v].opinion=0.5);
        nw[v].opinion=0;
      }
      if(nw[w].opinion!=0){
        BOOST_TEST(nw[w].opinion=0.5);
        nw[w].opinion=0;
      }
      if(nw[x].opinion!=1){
        BOOST_TEST(nw[w].opinion=0.5);
        nw[x].opinion=1;
      }
      if(nw[y].opinion!=1){
        BOOST_TEST(nw[w].opinion=0.5);
        nw[y].opinion=1;
      }

      revision::user_revision<reduced_s>(nw, true, 0.75, 2, uniform_prob_distr, rng);

       if(nw[v].opinion!=0){
         BOOST_TEST(nw[v].opinion=0.25);
         nw[v].opinion=0;
       }
       if(nw[w].opinion!=0){
         BOOST_TEST(nw[w].opinion=0.25);
         nw[w].opinion=0;
       }
       if(nw[x].opinion!=1){
         BOOST_TEST(nw[w].opinion=0.75);
         nw[x].opinion=1;
       }
       if(nw[y].opinion!=1){
         BOOST_TEST(nw[w].opinion=0.75);
         nw[y].opinion=1;
      }

      revision::user_revision<reduced_s>(nw, true, 0.25, 2, uniform_prob_distr, rng);

       if(nw[v].opinion!=0){
         BOOST_TEST(nw[v].opinion=0.75);
         nw[v].opinion=0;
       }
       if(nw[w].opinion!=0){
         BOOST_TEST(nw[w].opinion=0.75);
         nw[w].opinion=0;
       }
       if(nw[x].opinion!=1){
         BOOST_TEST(nw[w].opinion=0.25);
         nw[x].opinion=1;
       }
       if(nw[y].opinion!=1){
         BOOST_TEST(nw[w].opinion=0.25);
         nw[y].opinion=1;
      }
    }
}

BOOST_FIXTURE_TEST_CASE( test_conflict_dir,
                       TestNetwork2,
                       * boost::unit_test::tolerance(1e-12)) {

    {
      vertex v = 0;
      vertex w = 1;

      nw[v].group=0;
      nw[w].group=1;

      nw[v].discriminates = true;
      nw[w].discriminates = true;

      nw[v].opinion=0.3;
      nw[w].opinion=0.7;

      nw[v].tolerance=1;
      nw[w].tolerance=1;

      nw[v].susceptibility_1=0.5;
      nw[w].susceptibility_1=0;

      nw[v].susceptibility_2 = 0;
      nw[w].susceptibility_2 = 0.5;

      revision::user_revision<conflict_dir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==1.5/7);
      BOOST_TEST(nw[w].opinion==0.5);

      nw[v].opinion=0.2;
      revision::user_revision<conflict_dir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==0.1625);
      BOOST_TEST(nw[w].opinion==0.35);

      nw[v].opinion=0.2;
      nw[w].opinion=0.4;
      revision::user_revision<conflict_dir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==0.175);
      BOOST_TEST(nw[w].opinion==0.3);

      nw[v].opinion=0.5;
      nw[w].opinion=1;
      revision::user_revision<conflict_dir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==0.25);
      BOOST_TEST(nw[w].opinion==0.75);

      nw[v].opinion=0.75;
      revision::user_revision<conflict_dir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==nw[w].opinion);
      BOOST_TEST(nw[v].opinion==0.75);

      nw[v].opinion=0;
      nw[w].opinion=1;
      revision::user_revision<conflict_dir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==0.);
      BOOST_TEST(nw[w].opinion==0.5);

      nw[v].opinion=0.5;
      nw[w].opinion=0;
      revision::user_revision<conflict_dir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==0.75);
      BOOST_TEST(nw[w].opinion==0.25);

      revision::user_revision<conflict_dir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==5./6);
      BOOST_TEST(nw[w].opinion==0.5);

      nw[v].susceptibility_1=1;
      nw[w].opinion=0;
      revision::user_revision<conflict_dir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==1);
    }
}
BOOST_FIXTURE_TEST_CASE( test_conflict_undir,
                       TestNetwork2,
                       * boost::unit_test::tolerance(1e-12)) {
    {
      vertex v = 0;
      vertex w = 1;

      nw[v].group=0;
      nw[w].group=1;

      nw[v].discriminates = true;
      nw[w].discriminates = true;

      nw[v].opinion=0.4;
      nw[w].opinion=0.7;

      nw[v].tolerance=1;
      nw[w].tolerance=1;

      nw[v].susceptibility_1=0.5;
      nw[w].susceptibility_1=0.5;

      nw[v].susceptibility_2 = 0.;
      nw[w].susceptibility_2 = 0.;

      revision::user_revision<conflict_undir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==.3);
      BOOST_TEST(nw[w].opinion==0.7642857142857142);

      nw[v].opinion=0;
      nw[w].opinion=0.6;
      revision::user_revision<conflict_undir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==0);
      BOOST_TEST(nw[w].opinion==0.8);

      revision::user_revision<conflict_undir>(nw, true, 1, 2, uniform_prob_distr, rng);
      revision::user_revision<conflict_undir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==0);
      BOOST_TEST(nw[w].opinion==0.95);

      nw[w].opinion=1.;
      revision::user_revision<conflict_undir>(nw, true, 1, 2, uniform_prob_distr, rng);
      BOOST_TEST(nw[v].opinion==0);
      BOOST_TEST(nw[w].opinion==1.);

      nw[w].opinion=0;
      BOOST_TEST(nw[v].opinion==0);
      BOOST_TEST(nw[w].opinion==0.);
    }
}

} // namespace Utopia::Models::OpDisc
