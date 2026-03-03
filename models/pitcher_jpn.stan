// Japanese pitcher year-ahead ERA prediction
//
// Model: actual_ERA = Marcel_ERA + delta_K * z_K + delta_BB * z_BB + noise
//
// If delta_K = delta_BB = 0, reduces to pure Marcel (baseline).
// K% and BB% are z-scored using the training-set mean/sd.
// Negative delta_K:  high strikeout rate -> lower ERA (better)
// Positive delta_BB: high walk rate      -> higher ERA (worse)

data {
  int<lower=1> N;               // training observations
  vector[N] marcel_era;         // Marcel-projected ERA (prior mean)
  vector[N] z_K;                // z-scored K%  (SO / BF)
  vector[N] z_BB;               // z-scored BB% (BB / BF)
  vector[N] actual_era;         // actual next-year ERA (target)

  int<lower=0> N_pred;          // test observations
  vector[N_pred] marcel_era_pred;
  vector[N_pred] z_K_pred;
  vector[N_pred] z_BB_pred;
}

parameters {
  real delta_K;                 // K% correction weight
  real delta_BB;                // BB% correction weight
  real<lower=0> sigma;          // residual std
}

model {
  // Regularized priors: corrections start near zero
  delta_K  ~ normal(0, 0.5);
  delta_BB ~ normal(0, 0.5);
  sigma    ~ exponential(1);    // ERA residual scale ~1.0

  actual_era ~ normal(
    marcel_era + delta_K * z_K + delta_BB * z_BB,
    sigma
  );
}

generated quantities {
  vector[N_pred] stan_pred;
  vector[N] log_lik;

  for (i in 1:N_pred) {
    stan_pred[i] = marcel_era_pred[i]
                   + delta_K  * z_K_pred[i]
                   + delta_BB * z_BB_pred[i];
  }
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(
      actual_era[n] |
      marcel_era[n] + delta_K * z_K[n] + delta_BB * z_BB[n],
      sigma
    );
  }
}
