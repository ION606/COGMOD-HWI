data {
  int<lower=0> N;             // Number of training samples
  int<lower=0> M;             // Number of predictors (3: bmi, age, children)
  matrix[N, M] X;             // Training predictors
  vector[N] y;                // Training target (charges)
  int<lower=0> N_test;        // Number of test samples
  matrix[N_test, M] X_test;   // Test predictors
}
parameters {
  real alpha;                 // Intercept
  vector[M] beta;             // Regression coefficients
  real<lower=0> sigma;        // Noise term
}
model {
  // Priors
  sigma ~ inv_gamma(2, 2);
  alpha ~ normal(0, 10);
  beta ~ normal(0, 1);

  // Likelihood
  y ~ normal(alpha + X * beta, sigma);
}
generated quantities {
  vector[N_test] y_pred;      
  for (i in 1:N_test) {
    y_pred[i] = normal_rng(alpha + X_test[i] * beta, sigma);
  }
}