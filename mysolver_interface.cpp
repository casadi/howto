/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include "mysolver_interface.hpp"
#include <casadi/core/std_vector_tools.hpp>
#include <casadi/core/matrix/matrix_tools.hpp>
#include <casadi/core/mx/mx_tools.hpp>
#include <casadi/core/function/mx_function.hpp>
#include <ctime>
#include <cstring>

using namespace std;

namespace casadi {

  extern "C"
  int casadi_register_nlpsolver_mysolver(NlpSolverInternal::Plugin* plugin) {
    plugin->creator = MysolverInterface::creator;
    plugin->name = "mysolver";
    plugin->version = 23;
    return 0;
  }

  extern "C"
  void casadi_load_nlpsolver_mysolver() {
    NlpSolverInternal::registerPlugin(casadi_register_nlpsolver_mysolver);
  }

  MysolverInterface::MysolverInterface(const Function& nlp) : NlpSolverInternal(nlp) {

    addOption("the_answer_to_everything",             OT_REAL, 42);
    std::cout << "Construct" << std::endl;
  }

  MysolverInterface::~MysolverInterface() {

  }

  MysolverInterface* MysolverInterface::clone() const {
    // Use default copy routine
    MysolverInterface* node = new MysolverInterface(*this);
  }

  void MysolverInterface::init() {
    // Call the init method of the base class
    NlpSolverInternal::init();

    std::cout << "Init" << std::endl;

    the_answer_ = getOption("the_answer_to_everything");

  }

  void MysolverInterface::evaluate() {
    log("MysolverInterface::evaluate");

    output(NLP_SOLVER_X).set(the_answer_);
    
  }

  bool MysolverInterface::eval_h(const double* x, bool new_x, double obj_factor,
                             const double* lambda, bool new_lambda, int nele_hess,
                             int* iRow, int* jCol, double* values) {
    try {
      log("eval_h started");

      if (values == NULL) {
        int nz=0;
        const int* colind = hessLag_.output().colind();
        int ncol = hessLag_.output().size2();
        const int* row = hessLag_.output().row();
        for (int cc=0; cc<ncol; ++cc)
          for (int el=colind[cc]; el<colind[cc+1] && row[el]<=cc; ++el) {
            iRow[nz] = row[el];
            jCol[nz] = cc;
            nz++;
          }
      } else {
        // Pass the argument to the function
        hessLag_.setInput(x, NL_X);
        hessLag_.setInput(input(NLP_SOLVER_P), NL_P);
        hessLag_.setInput(obj_factor, NL_NUM_IN+NL_F);
        hessLag_.setInput(lambda, NL_NUM_IN+NL_G);

        // Evaluate
        hessLag_.evaluate();

        // Get results
        hessLag_.output().getSym(values);

      }

      log("eval_h ok");
      return true;
    } catch(exception& ex) {
      if (eval_errors_fatal_) throw ex;
      cerr << "eval_h failed: " << ex.what() << endl;
      return false;
    }
  }

  bool MysolverInterface::eval_jac_g(int n, const double* x, bool new_x, int m, int nele_jac,
                                  int* iRow, int *jCol, double* values) {
    try {
      log("eval_jac_g started");

      // Quich finish if no constraints
      if (m==0) {
        log("eval_jac_g quick return (m==0)");
        return true;
      }

      // Get function
      Function& jacG = this->jacG();

      if (values == NULL) {
        int nz=0;
        const int* colind = jacG.output().colind();
        int ncol = jacG.output().size2();
        const int* row = jacG.output().row();
        for (int cc=0; cc<ncol; ++cc)
          for (int el=colind[cc]; el<colind[cc+1]; ++el) {
            int rr = row[el];
            iRow[nz] = rr;
            jCol[nz] = cc;
            nz++;
          }
      } else {
        // Pass the argument to the function
        jacG.setInput(x, NL_X);
        jacG.setInput(input(NLP_SOLVER_P), NL_P);

        // Evaluate the function
        jacG.evaluate();

        // Get the output
        jacG.getOutput(values);

      }

      log("eval_jac_g ok");
      return true;
    } catch(exception& ex) {
      if (eval_errors_fatal_) throw ex;
      cerr << "eval_jac_g failed: " << ex.what() << endl;
      return false;
    }
  }

  bool MysolverInterface::eval_f(int n, const double* x, bool new_x, double& obj_value) {
    try {
      log("eval_f started");

      casadi_assert(n == nx_);

      // Pass the argument to the function
      nlp_.setInput(x, NL_X);
      nlp_.setInput(input(NLP_SOLVER_P), NL_P);

      // Evaluate the function
      nlp_.evaluate();

      // Get the result
      nlp_.getOutput(obj_value, NL_F);

      log("eval_f ok");
      return true;
    } catch(exception& ex) {
      if (eval_errors_fatal_) throw ex;
      cerr << "eval_f failed: " << ex.what() << endl;
      return false;
    }

  }

  bool MysolverInterface::eval_g(int n, const double* x, bool new_x, int m, double* g) {
    try {
      log("eval_g started");

      if (m>0) {
        // Pass the argument to the function
        nlp_.setInput(x, NL_X);
        nlp_.setInput(input(NLP_SOLVER_P), NL_P);

        // Evaluate the function and tape
        nlp_.evaluate();

        // Ge the result
        nlp_.getOutput(g, NL_G);

      }

      log("eval_g ok");
      return true;
    } catch(exception& ex) {
      if (eval_errors_fatal_) throw ex;
      cerr << "eval_g failed: " << ex.what() << endl;
      return false;
    }
  }

  bool MysolverInterface::eval_grad_f(int n, const double* x, bool new_x, double* grad_f) {
    try {
      log("eval_grad_f started");
      casadi_assert(n == nx_);

      // Pass the argument to the function
      gradF_.setInput(x, NL_X);
      gradF_.setInput(input(NLP_SOLVER_P), NL_P);

      // Evaluate, adjoint mode
      gradF_.evaluate();

      // Get the result
      gradF_.output().get(grad_f);

      log("eval_grad_f ok");
      return true;
    } catch(exception& ex) {
      if (eval_errors_fatal_) throw ex;
      cerr << "eval_grad_f failed: " << ex.what() << endl;
      return false;
    }
  }


} // namespace casadi
