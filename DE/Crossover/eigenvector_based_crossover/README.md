source: An efficient eigenvector-based crossover for differential evolution Simplifying with rank-one updates

example: 
  eig_P = 0.5; 
  P = InitializePop(...); 
  params_eig = EigenCovInitialization(P);
  for ... 
    V = Mutation(...); 
    T = EigenvectorCrossover(P,V,CR,eig_P,eig.C); 
    P_new,P_fit_new = Selection(...) 
    params_eig = EigenCovUpdate(P_new, P_fit_new, params_eig)
