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


#ifndef CASADI_MYSOLVER_INTERFACE_HPP
#define CASADI_MYSOLVER_INTERFACE_HPP

#include <casadi/core/function/nlp_solver_internal.hpp>

namespace casadi {


  class MysolverInterface : public NlpSolverInternal {

  public:
    // Constructor
    explicit MysolverInterface(const Function& nlp);

    // Destructor
    virtual ~MysolverInterface();

    // Clone function
    virtual MysolverInterface* clone() const;

    /** \brief  Create a new NLP Solver */
    static NlpSolverInternal* creator(const Function& nlp)
    { return new MysolverInterface(nlp);}

    // (Re)initialize
    virtual void init();

    // Solve the NLP
    virtual void evaluate();

    // Callback functions
    bool eval_f(int n, const double* x, bool new_x, double& obj_value);
    bool eval_grad_f(int n, const double* x, bool new_x, double* grad_f);
    bool eval_g(int n, const double* x, bool new_x, int m, double* g);
    bool eval_jac_g(int n, const double* x, bool new_x, int m, int nele_jac, int* iRow, int *jCol,
                    double* values);
    bool eval_h(const double* x, bool new_x, double obj_factor, const double* lambda,
                bool new_lambda, int nele_hess, int* iRow, int* jCol, double* values);

  private:
    double the_answer_;

  };

} // namespace casadi

/// \endcond
#endif // CASADI_MYSOLVER_INTERFACE_HPP
