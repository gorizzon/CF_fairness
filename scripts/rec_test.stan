data {
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of sensitive covariates (male-female here)
  matrix[N, K]   a; // sensitive variables
  real          grade[N];         // ind.university_grade
  int           debate[N];        // ind.debateclub
  int           programming[N];   // ind.programming_exp
  int           internat[N];      // ind.international_exp
  int           entrep[N];        // ind.entrepeneur_exp
  int           lang[N];          // ind.languages
  int           study[N];         // ind.exact_study
  int           degree[N];        // ind.degree
  real          grade0;
  real          eta_u_grade;
  vector[K]     eta_a_grade;
  real          debate0;
  real          eta_u_debate;
  vector[K]     eta_a_debate;
  real          programming0;
  real          eta_u_programming;
  vector[K]     eta_a_programming;
  real          internat0;
  real          eta_u_internat;
  vector[K]     eta_a_internat;
  real          entrep0;
  real          eta_u_entrep;
  vector[K]     eta_a_entrep;
  real          lang0;
  real          eta_u_lang;
  vector[K]     eta_a_lang;
  real          study0;
  real          eta_u_study;
  vector[K]     eta_a_study;
  real          degree0;
  real          eta_u_degree;
  vector[K]     eta_a_degree;
  real          sigma_g;
}

parameters {
  vector[N] u;
}

model {
  // Prior distribution
  u		~ normal(0, 1);

  // Likelihood for continuous variable
  grade		~ normal(grade0 + eta_u_grade * u + a * eta_a_grade, sigma_g);

  // Likelihood for categorical and binary variables
  for (n in 1:N) {
    debate[n]        ~ bernoulli_logit(debate0 + eta_u_debate * u[n] + a[n,] * eta_a_debate);
    programming[n]   ~ bernoulli_logit(programming0 + eta_u_programming * u[n] + a[n,] * eta_a_programming);
    internat[n]      ~ bernoulli_logit(internat0 + eta_u_internat * u[n] + a[n,] * eta_a_internat);
    entrep[n]        ~ bernoulli_logit(entrep0 + eta_u_entrep * u[n] + a[n,] * eta_a_entrep);
    lang[n]          ~ poisson_log(lang0 + eta_u_lang * u[n] + a[n,] * eta_a_lang);
    study[n]         ~ bernoulli_logit(study0 + eta_u_study * u[n] + a[n,] * eta_a_study);
    degree[n]        ~ poisson_log(degree0 + eta_u_degree * u[n] + a[n,] * eta_a_degree);
  }

}
