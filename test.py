from casadi import *

x = SX.sym("x")

nlp = SXFunction(nlpIn(x=x),nlpOut(f=x**2))
nlp.init()

solver = NlpSolver("mysolver",nlp)
solver.init()

solver.evaluate()

print "answer:", solver.getOutput("x")
