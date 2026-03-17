source: An efficient eigenvector-based crossover for differential evolution Simplifying with rank-one updates

example: 
  eig_P = 0.5; 
  P = InitializePop(...); 
  params_eig = InitializeEigenCov(P);
  for ... 
    V = Mutation(...); 
    T = Crossover_Eigenvector(P,V,CR,eig_P,eig.C); 
    P_new,P_fit_new = Selection(...) 
    params_eig = UpdateEigenCov(P_new, P_fit_new, params_eig)
