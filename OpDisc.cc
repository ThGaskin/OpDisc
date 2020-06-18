#include <iostream>

#include "OpDisc.hh"

using namespace Utopia::Models::OpDisc;

int main (int, char** argv)
{
    try{
        // Initialize the PseudoParent from config file path
        Utopia::PseudoParent pp(argv[1]);

        auto model_cfg = pp.get_cfg()["OpDisc"];
        auto mode = Utopia::get_as<std::string>("mode", model_cfg);
        if (mode=="conflict_dir"){
            OpDisc<conflict_dir> model("OpDisc", pp);
            model.run();
        }
        else if (mode=="conflict_undir"){
            OpDisc<conflict_undir> model("OpDisc", pp);
            model.run();
        }
        else if (mode=="isolated_1"){
            OpDisc<isolated_1> model("OpDisc", pp);
            model.run();
        }
        else if (mode=="isolated_2"){
            OpDisc<isolated_2> model("OpDisc", pp);
            model.run();
        }
        else if (mode=="reduced_int_prob"){
            OpDisc<reduced_int_prob> model("OpDisc", pp);
            model.run();
        }
        else if (mode=="reduced_s"){
            OpDisc<reduced_s> model("OpDisc", pp);
            model.run();
        }
        else{
            throw std::invalid_argument("Mode {} unknown!");
        }
        return 0;
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Exception occured!" << std::endl;
        return 1;
    }
}
