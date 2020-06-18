#define BOOST_TEST_MODULE test init

#include <boost/test/unit_test.hpp>

#include <utopia/core/model.hh>

#include "../modes.hh"
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
        const int num_vertices=get_as<int>("num_users", cfg);
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

BOOST_FIXTURE_TEST_CASE( test_init_Gauss_2_groups,
                         TestNetwork,
                         * boost::unit_test::tolerance(1e-12))
{
    const int num_groups = 2;

    // Test initialisation functions .........................................
    {
        unsigned int i = 0;

        double op_avg=0.;
        double op_avg_group_1=0;
        double op_avg_group_2=0;

        std::vector<double> group_1;
        std::vector<double> group_2;


        for (auto v : range<IterateOver::vertices>(nw)) {

            nw[v].tolerance=utils::tolerance_func(.6, .2);
            BOOST_TEST(nw[v].tolerance == .196);

            nw[v].tolerance=utils::tolerance_func(0, .05);
            BOOST_TEST(nw[v].tolerance == .025);

            nw[v].tolerance=utils::tolerance_func(0, .2);
            BOOST_TEST(nw[v].tolerance == .1);

            int q = num_groups;
            if (num_groups>2){q-=1;}
            nw[v].group=i%q;
            ++i;
            if (num_groups>2 and nw[v].group==0){
                nw[v].group=(num_groups-1)*utils::rand_int(0, 1, rng);
            }

            nw[v].opinion=utils::initialize_op<reduced_s>(num_groups, nw[v].group, rng);
            op_avg+=nw[v].opinion;
            BOOST_TEST(0<=nw[v].opinion);
            BOOST_TEST(nw[v].opinion<=1);

            BOOST_TEST(0<=nw[v].group);
            BOOST_TEST(nw[v].group<=num_groups);
            if(nw[v].group==0){
              group_1.push_back(1);
              op_avg_group_1+=nw[v].opinion;
            }
            else if(nw[v].group==1){
              group_2.push_back(1);
              op_avg_group_2+=nw[v].opinion;
            }

        }
        BOOST_TEST(group_1.size()==boost::num_vertices(nw)/num_groups);
        BOOST_TEST(group_2.size()==boost::num_vertices(nw)/num_groups);
        BOOST_TEST(op_avg/boost::num_vertices(nw)==0.5, boost::test_tools::tolerance(0.01));
        BOOST_TEST(op_avg_group_1/group_1.size()<0.5);

    }
}

BOOST_FIXTURE_TEST_CASE( test_init_Gauss_5_groups,
                         TestNetwork,
                         * boost::unit_test::tolerance(1e-12))
{
    const int num_groups = 5;

    // Test initialisation functions .........................................
    {
        unsigned int i = 0;

        double op_avg=0.;
        double op_avg_group_1=0;
        double op_avg_group_2=0;
        double op_avg_group_3=0;
        double op_avg_group_4=0;
        double op_avg_group_5=0;
        std::vector<double> group_1;
        std::vector<double> group_2;
        std::vector<double> group_3;
        std::vector<double> group_4;
        std::vector<double> group_5;

        for (auto v : range<IterateOver::vertices>(nw)) {

            nw[v].tolerance=utils::tolerance_func(.5, .3);
            BOOST_TEST(nw[v].tolerance == .3);

            nw[v].tolerance=utils::tolerance_func(.1, .4);
            BOOST_TEST(nw[v].tolerance == .272);

            nw[v].tolerance=utils::tolerance_func(1, .1);
            BOOST_TEST(nw[v].tolerance == .05);

            int q = num_groups;
            if (num_groups>2){q-=1;}
            nw[v].group=i%q;
            ++i;
            if (num_groups>2 and nw[v].group==0){
                nw[v].group=(num_groups-1)*utils::rand_int(0, 1, rng);
            }

            nw[v].opinion=utils::initialize_op<reduced_s>(num_groups, nw[v].group, rng);
            op_avg+=nw[v].opinion;
            BOOST_TEST(0<=nw[v].opinion);
            BOOST_TEST(nw[v].opinion<=1);

            BOOST_TEST(0<=nw[v].group);
            BOOST_TEST(nw[v].group<num_groups);
            if(nw[v].group==0){
              group_1.push_back(1);
              op_avg_group_1+=nw[v].opinion;
            }
            else if(nw[v].group==1){
              group_2.push_back(1);
              op_avg_group_2+=nw[v].opinion;
            }
            else if(nw[v].group==2){
              group_3.push_back(1);
              op_avg_group_3+=nw[v].opinion;
            }
            else if(nw[v].group==3){
              group_4.push_back(1);
              op_avg_group_4+=nw[v].opinion;
            }
            else if(nw[v].group==4){
              group_5.push_back(1);
              op_avg_group_5+=nw[v].opinion;
            }

        }
        BOOST_TEST(group_1.size()+group_5.size()==boost::num_vertices(nw)/(num_groups-1));
        BOOST_TEST(group_2.size()<=boost::num_vertices(nw)/(num_groups-1));
        BOOST_TEST(group_3.size()<=boost::num_vertices(nw)/(num_groups-1));
        BOOST_TEST(group_4.size()<=boost::num_vertices(nw)/(num_groups-1));

        BOOST_TEST(op_avg/boost::num_vertices(nw)==0.5, boost::test_tools::tolerance(0.01));
        BOOST_TEST(op_avg_group_1/group_1.size()<op_avg_group_2/group_2.size());
        BOOST_TEST(op_avg_group_2/group_2.size()==1./(num_groups-1), boost::test_tools::tolerance(0.05));
        BOOST_TEST(op_avg_group_3/group_3.size()==2./(num_groups-1), boost::test_tools::tolerance(0.05));
        BOOST_TEST(op_avg_group_4/group_4.size()==3./(num_groups-1), boost::test_tools::tolerance(0.05));
        BOOST_TEST(op_avg_group_5/group_5.size()>op_avg_group_4/group_4.size());
    }
}


BOOST_FIXTURE_TEST_CASE( test_init_unif,
                         TestNetwork,
                         * boost::unit_test::tolerance(1e-12))
{
    const int num_groups = 6;

    // Test initialisation functions .........................................
    {
        unsigned int i = 0;

        double op_avg=0.;
        double op_avg_group_1=0;
        double op_avg_group_2=0;
        double op_avg_group_3=0;
        double op_avg_group_4=0;
        double op_avg_group_5=0;
        double op_avg_group_6=0;

        std::vector<double> group_1;
        std::vector<double> group_2;
        std::vector<double> group_3;
        std::vector<double> group_4;
        std::vector<double> group_5;
        std::vector<double> group_6;

        for (auto v : range<IterateOver::vertices>(nw)) {

            nw[v].tolerance=utils::tolerance_func(.5, .3);
            BOOST_TEST(nw[v].tolerance == .3);

            nw[v].tolerance=utils::tolerance_func(.1, .4);
            BOOST_TEST(nw[v].tolerance == .272);

            nw[v].tolerance=utils::tolerance_func(1, .1);
            BOOST_TEST(nw[v].tolerance == .05);

            int q = num_groups;
            if (num_groups>2){q-=1;}
            nw[v].group=i%q;
            ++i;
            if (num_groups>2 and nw[v].group==0){
                nw[v].group=(num_groups-1)*utils::rand_int(0, 1, rng);
            }

            nw[v].opinion=utils::initialize_op<conflict_dir>(num_groups, nw[v].group, rng);
            op_avg+=nw[v].opinion;
            BOOST_TEST(0<=nw[v].opinion);
            BOOST_TEST(nw[v].opinion<=1);

            BOOST_TEST(0<=nw[v].group);
            BOOST_TEST(nw[v].group<=num_groups);
            if(nw[v].group==0){
              group_1.push_back(1);
              op_avg_group_1+=nw[v].opinion;
            }
            else if(nw[v].group==1){
              group_2.push_back(1);
              op_avg_group_2+=nw[v].opinion;
            }
            else if(nw[v].group==2){
              group_3.push_back(1);
              op_avg_group_3+=nw[v].opinion;
            }
            else if(nw[v].group==3){
              group_4.push_back(1);
              op_avg_group_4+=nw[v].opinion;
            }
            else if(nw[v].group==4){
              group_5.push_back(1);
              op_avg_group_5+=nw[v].opinion;
            }
            else if(nw[v].group==5){
              group_6.push_back(1);
              op_avg_group_6+=nw[v].opinion;
            }

        }
        BOOST_TEST(group_1.size()+group_6.size()==boost::num_vertices(nw)/(num_groups-1));
        BOOST_TEST(group_2.size()==boost::num_vertices(nw)/(num_groups-1));
        BOOST_TEST(group_3.size()==boost::num_vertices(nw)/(num_groups-1));
        BOOST_TEST(group_4.size()==boost::num_vertices(nw)/(num_groups-1));
        BOOST_TEST(group_5.size()==boost::num_vertices(nw)/(num_groups-1));

        BOOST_TEST(op_avg/boost::num_vertices(nw)==0.5, boost::test_tools::tolerance(0.01));
        BOOST_TEST(op_avg_group_1/group_1.size()==0.5, boost::test_tools::tolerance(0.05));
        BOOST_TEST(op_avg_group_2/group_2.size()==0.5, boost::test_tools::tolerance(0.05));
        BOOST_TEST(op_avg_group_3/group_3.size()==0.5, boost::test_tools::tolerance(0.05));
        BOOST_TEST(op_avg_group_4/group_4.size()==0.5, boost::test_tools::tolerance(0.05));
        BOOST_TEST(op_avg_group_5/group_5.size()==0.5, boost::test_tools::tolerance(0.05));
        BOOST_TEST(op_avg_group_6/group_6.size()==0.5, boost::test_tools::tolerance(0.05));

    }
}


} // namespace Utopia::Models::OpDisc
