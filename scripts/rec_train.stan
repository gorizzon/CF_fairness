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
  int           decision[N];      // decision
}

transformed data {
  vector[K] zero_K;
  vector[K] one_K;

  zero_K = rep_vector(0, K);
  one_K = rep_vector(1, K);
}

parameters {
  vector[N] u;

  real grade0;
  real eta_u_grade;
  real debate0;
  real eta_u_debate;
  real programming0;
  real eta_u_programming;
  real internat0;
  real eta_u_internat;
  real entrep0;
  real eta_u_entrep;
  real lang0;
  real eta_u_lang;
  real study0;
  real eta_u_study;
  real degree0;
  real eta_u_degree;
  real eta_u_decision;

  vector[K] eta_a_grade;
  vector[K] eta_a_debate;
  vector[K] eta_a_programming;
  vector[K] eta_a_internat;
  vector[K] eta_a_entrep;
  vector[K] eta_a_lang;
  vector[K] eta_a_study;
  vector[K] eta_a_degree;
  vector[K] eta_a_decision;

  real<lower=0> sigma_g_Sq;
}

transformed parameters {
  // Population standard deviation (a positive real number)
  real<lower=0> sigma_g;
  // Standard deviation (derived from variance)
  sigma_g = sqrt(sigma_g_Sq);
}

model {
  // Prior distributions
  u ~ normal(0, 1);

  grade0          ~ normal(0, 1);
  eta_u_grade     ~ normal(0, 1);
  debate0         ~ normal(0, 1);
  eta_u_debate    ~ normal(0, 1);
  programming0    ~ normal(0, 1);
  eta_u_programming ~ normal(0, 1);
  internat0       ~ normal(0, 1);
  eta_u_internat  ~ normal(0, 1);
  entrep0         ~ normal(0, 1);
  eta_u_entrep    ~ normal(0, 1);
  lang0           ~ normal(0, 1);
  eta_u_lang      ~ normal(0, 1);
  study0          ~ normal(0, 1);
  eta_u_study     ~ normal(0, 1);
  degree0         ~ normal(0, 1);
  eta_u_degree    ~ normal(0, 1);
  eta_u_decision  ~ normal(0, 1);

  eta_a_grade     ~ normal(0, 1);
  eta_a_debate    ~ normal(0, 1);
  eta_a_programming ~ normal(0, 1);
  eta_a_internat  ~ normal(0, 1);
  eta_a_entrep    ~ normal(0, 1);
  eta_a_lang      ~ normal(0, 1);
  eta_a_study     ~ normal(0, 1);
  eta_a_degree    ~ normal(0, 1);
  eta_a_decision  ~ normal(0, 1);

  sigma_g_Sq      ~ inv_gamma(1, 1);

  // Likelihood for continuous variable
  grade ~ normal(grade0 + eta_u_grade * u + a * eta_a_grade, sigma_g);

  // Likelihood for binary and count variables
  for (n in 1:N) {
    debate[n]        ~ bernoulli_logit(debate0 + eta_u_debate * u[n] + a[n,] * eta_a_debate);
    programming[n]   ~ bernoulli_logit(programming0 + eta_u_programming * u[n] + a[n,] * eta_a_programming);
    internat[n]      ~ bernoulli_logit(internat0 + eta_u_internat * u[n] + a[n,] * eta_a_internat);
    entrep[n]        ~ bernoulli_logit(entrep0 + eta_u_entrep * u[n] + a[n,] * eta_a_entrep);
    lang[n]          ~ poisson_log(lang0 + eta_u_lang * u[n] + a[n,] * eta_a_lang);
    study[n]         ~ bernoulli_logit(study0 + eta_u_study * u[n] + a[n,] * eta_a_study);
    degree[n]        ~ poisson_log(degree0 + eta_u_degree * u[n] + a[n,] * eta_a_degree);
    decision[n]      ~ bernoulli_logit(eta_u_decision * u[n] + a[n,] * eta_a_decision);

  }
}
