# When Does Hitting the Ball Harder Matter?
# 2026 Statcast Analysis

# Load packages

packages <- c("readr", "dplyr", "ggplot2", "glmnet",
              "tidyr", "scales", "tibble")

missing_packages <- packages[
  !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Install these packages first: install.packages(c(",
    paste(sprintf('"%s"', missing_packages), collapse = ", "),
    "))"
  )
}

library(readr)
library(dplyr)
library(ggplot2)
library(glmnet)

# Import Statcast data

months <- c(
  "March-April", "May", "June",
  "July", "August", "September"
)

paths <- file.path("statcast_data", paste0(months, ".csv"))

if (any(!file.exists(paths))) {
  stop(
    "Missing Statcast file(s): ",
    paste(paths[!file.exists(paths)], collapse = ", "),
    ". Check the statcast_data folder."
  )
}

dir.create("outputs", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

keep <- c(
  "game_date", "game_pk", "batter",
  "home_team", "away_team", "inning_topbot",
  "launch_speed", "launch_angle", "events",
  "estimated_ba_using_speedangle"
)

message("Reading the six monthly datasets...")

raw <- bind_rows(lapply(paths, function(f) {
  x <- read_csv(f, show_col_types = FALSE, progress = FALSE)
  select(x, all_of(keep))
}))

# Clean the dataset and identify hits.
# The hit rate here only includes tracked batted balls.

d <- raw |>
  filter(
    !is.na(launch_speed),
    !is.na(launch_angle),
    !is.na(events)
  ) |>
  mutate(
    date = as.Date(game_date),
    hit = as.integer(
      events %in% c("single", "double", "triple", "home_run")
    ),
    batting_team = if_else(
      inning_topbot == "Top",
      away_team,
      home_team
    )
  )

cat(sprintf(
  "Records: %s raw, %s complete\n",
  nrow(raw), nrow(d)
))

# Create the predictor variables for the models.
# The nonlinear models include squared and cubic terms.

make_features <- function(df, include_angle, degree) {
  
  x <- df$launch_speed
  
  if (!include_angle) {
    
    if (degree == 1) {
      return(cbind(ev = x))
    }
    
    return(cbind(
      ev = x,
      ev2 = x^2,
      ev3 = x^3
    ))
  }
  
  a <- df$launch_angle
  
  if (degree == 1) {
    return(cbind(ev = x, angle = a))
  }
  
  cbind(
    ev = x,
    angle = a,
    ev2 = x^2,
    ev_angle = x * a,
    angle2 = a^2,
    ev3 = x^3,
    ev2_angle = x^2 * a,
    ev_angle2 = x * a^2,
    angle3 = a^3
  )
}

# Fit each logistic regression model.
# The simple baseline uses glm().
# The other models use ridge regression.

fit_model <- function(training, include_angle, degree) {
  
  if (!include_angle && degree == 1) {
    
    m <- glm(
      hit ~ launch_speed,
      data = training,
      family = binomial()
    )
    
    return(list(
      model = m,
      simple = TRUE,
      include_angle = include_angle,
      degree = degree
    ))
  }
  
  x <- make_features(training, include_angle, degree)
  
  centers <- colMeans(x)
  
  spreads <- sqrt(
    colMeans(sweep(x, 2, centers, "-")^2)
  )
  
  if (any(!is.finite(spreads)) || any(spreads == 0)) {
    stop("Found a zero-variance or invalid predictor")
  }
  
  xs <- sweep(
    sweep(x, 2, centers, "-"),
    2, spreads, "/"
  )
  
  m <- glmnet(
    xs,
    training$hit,
    family = "binomial",
    alpha = 0,
    lambda = 1 / nrow(training),
    standardize = FALSE,
    intercept = TRUE,
    control = list(
      thresh = 1e-10,
      maxit = 100000
    )
  )
  
  list(
    model = m,
    simple = FALSE,
    centers = centers,
    spreads = spreads,
    include_angle = include_angle,
    degree = degree
  )
}

# Generate predicted hit probabilities.

predict_model <- function(fit, new_data) {
  
  if (fit$simple) {
    return(as.numeric(
      predict(
        fit$model,
        newdata = new_data,
        type = "response"
      )
    ))
  }
  
  x <- make_features(
    new_data,
    fit$include_angle,
    fit$degree
  )
  
  xs <- sweep(
    sweep(x, 2, fit$centers, "-"),
    2, fit$spreads, "/"
  )
  
  as.numeric(
    predict(
      fit$model,
      newx = xs,
      type = "response"
    )
  )
}

# Calculate model performance.
# Lower Brier scores and log loss are better.
# Higher AUC is better.

auc_score <- function(y, p) {
  
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  
  if (n1 == 0 || n0 == 0) {
    return(NA_real_)
  }
  
  (
    sum(rank(p, ties.method = "average")[y == 1]) -
      n1 * (n1 + 1) / 2
  ) / (n1 * n0)
}

score <- function(y, p) {
  
  eps <- 1e-15
  pc <- pmin(pmax(p, eps), 1 - eps)
  
  c(
    brier = mean((p - y)^2),
    log_loss = -mean(
      y * log(pc) + (1 - y) * log(1 - pc)
    ),
    auc = auc_score(y, p)
  )
}

# Set up the four models.

specs <- data.frame(
  model = c(
    "EV only (linear)",
    "EV only (nonlinear degree 3)",
    "EV + angle (linear)",
    "EV + angle (nonlinear degree 3)"
  ),
  include_angle = c(FALSE, FALSE, TRUE, TRUE),
  degree = c(1, 3, 1, 3)
)

# Test the models separately on July, August, and September.
# Each test only uses earlier months for training.

rolling <- list()

for (test_month in 7:9) {
  
  train <- d |>
    filter(as.integer(format(date, "%m")) < test_month)
  
  test <- d |>
    filter(as.integer(format(date, "%m")) == test_month)
  
  cat(sprintf(
    "Validating month %s: train=%d, test=%d\n",
    month.name[test_month],
    nrow(train),
    nrow(test)
  ))
  
  for (i in seq_len(nrow(specs))) {
    
    fit <- fit_model(
      train,
      specs$include_angle[i],
      specs$degree[i]
    )
    
    p <- predict_model(fit, test)
    sc <- score(test$hit, p)
    
    rolling[[length(rolling) + 1]] <- data.frame(
      month = month.name[test_month],
      model = specs$model[i],
      n_train = nrow(train),
      n_test = nrow(test),
      brier = unname(sc["brier"]),
      log_loss = unname(sc["log_loss"]),
      auc = unname(sc["auc"])
    )
  }
}

validation <- bind_rows(rolling)

write_csv(
  validation,
  "outputs/rolling_month_validation_R.csv"
)

print(validation)

# Use March-August to predict September.

train <- d |>
  filter(date < as.Date("2026-09-01"))

test <- d |>
  filter(date >= as.Date("2026-09-01"))

for (i in seq_len(nrow(specs))) {
  
  fit <- fit_model(
    train,
    specs$include_angle[i],
    specs$degree[i]
  )
  
  test[[paste0("prediction_", i)]] <-
    predict_model(fit, test)
}

# Compare our models with Statcast expected batting average.

xba_test <- test |>
  filter(
    !is.na(estimated_ba_using_speedangle),
    between(estimated_ba_using_speedangle, 0, 1)
  )

xba_scores <- bind_rows(
  data.frame(
    method = "Statcast xBA",
    n = nrow(xba_test),
    t(score(
      xba_test$hit,
      xba_test$estimated_ba_using_speedangle
    ))
  ),
  bind_rows(lapply(1:4, function(i) {
    
    data.frame(
      method = specs$model[i],
      n = nrow(xba_test),
      t(score(
        xba_test$hit,
        xba_test[[paste0("prediction_", i)]]
      ))
    )
  }))
)

write_csv(
  xba_scores,
  "outputs/xba_comparison_R.csv"
)

print(xba_scores)

# Check whether predicted probabilities match actual hit rates.

calibration <- bind_rows(lapply(1:4, function(i) {
  
  col <- paste0("prediction_", i)
  
  tibble(
    model = specs$model[i],
    hit = test$hit,
    predicted = test[[col]]
  ) |>
    mutate(
      bin = cut(
        predicted,
        breaks = seq(0, 1, 0.1),
        include.lowest = TRUE
      )
    ) |>
    group_by(model, bin) |>
    summarise(
      n = n(),
      mean_prediction = mean(predicted),
      observed_hit_rate = mean(hit),
      .groups = "drop"
    )
}))

write_csv(
  calibration,
  "outputs/september_calibration_R.csv"
)

# Summarize the 2026 Mets hitters.
# Include players with at least 100 tracked batted balls
# in the main full-season comparison.

mets_all <- d |>
  filter(batting_team == "NYM")

full_season <- mets_all |>
  group_by(batter) |>
  summarise(
    n = n(),
    hits = sum(hit),
    hit_rate = mean(hit),
    average_ev = mean(launch_speed),
    average_angle = mean(launch_angle),
    games = n_distinct(game_pk),
    .groups = "drop"
  ) |>
  mutate(
    qualifies_100 = n >= 100
  ) |>
  arrange(desc(n))

# Match Mets batter IDs to player names.

player_key <- tibble::tibble(
  batter = c(
    665742, 666182, 596019, 701807,
    543760, 805999, 683146, 682626,
    676724, 668901, 620443, 673357,
    621438, 593871, 703492
  ),
  player = c(
    "Juan Soto",
    "Bo Bichette",
    "Francisco Lindor",
    "Carson Benge",
    "Marcus Semien",
    "A.J. Ewing",
    "Brett Baty",
    "Francisco Alvarez",
    "Jared Young",
    "Mark Vientos",
    "Luis Torrens",
    "Luis Robert Jr.",
    "Tyrone Taylor",
    "Jorge Polanco",
    "Nick Morabito"
  )
)

add_names <- function(tbl) {
  
  tbl |>
    left_join(player_key, by = "batter") |>
    mutate(
      player = if_else(
        is.na(player),
        paste0("ID ", batter, " (verify name)"),
        player
      )
    )
}

full_season <- add_names(full_season)

write_csv(
  full_season,
  "outputs/mets_full_season_R.csv"
)

# Compare actual and predicted September hits for Mets hitters.

mets_sep <- test |>
  filter(batting_team == "NYM")

sep_players <- mets_sep |>
  group_by(batter) |>
  summarise(
    n = n(),
    hits = sum(hit),
    expected_hits = sum(prediction_4),
    observed_rate = mean(hit),
    predicted_rate = mean(prediction_4),
    variance_hits = sum(
      prediction_4 * (1 - prediction_4)
    ),
    games = n_distinct(game_pk),
    .groups = "drop"
  ) |>
  mutate(
    actual_minus_expected = hits - expected_hits,
    diff_pp = 100 * (observed_rate - predicted_rate),
    approximate_z = actual_minus_expected /
      sqrt(variance_hits),
    qualifies_20 = n >= 20
  ) |>
  add_names() |>
  arrange(desc(n))

write_csv(
  sep_players,
  "outputs/mets_september_R.csv"
)

cat(sprintf(
  "Mets September: %d batted balls, %d actual hits, %.1f expected hits\n",
  nrow(mets_sep),
  sum(mets_sep$hit),
  sum(mets_sep$prediction_4)
))

print(
  sep_players |>
    filter(qualifies_20) |>
    select(
      player, n, hits, expected_hits,
      actual_minus_expected, approximate_z
    )
)

# Graph 1: Model performance across July-September.

chart_data <- validation |>
  filter(model %in% c(
    "EV only (nonlinear degree 3)",
    "EV + angle (nonlinear degree 3)"
  )) |>
  mutate(
    month = factor(
      month,
      levels = c("July", "August", "September")
    )
  )

p1 <- ggplot(
  chart_data,
  aes(x = month, y = brier, fill = model)
) +
  geom_col(
    position = position_dodge(width = 0.7),
    width = 0.6
  ) +
  labs(
    title = "Predicting hits on batted balls",
    subtitle = "Held-out monthly tests: lower Brier scores are better",
    x = "Test month",
    y = "Brier score",
    fill = "Model"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  "figures/model_validation.png",
  p1,
  width = 9,
  height = 5,
  dpi = 220
)

# Graph 2: Difference between actual and predicted hit rates.

p2_data <- sep_players |>
  filter(
    qualifies_20,
    !grepl("verify name", player)
  ) |>
  mutate(
    player_label = paste0(player, " (n=", n, ")"),
    difference = 100 * (observed_rate - predicted_rate)
  )

p2 <- ggplot(
  p2_data,
  aes(
    x = reorder(player_label, difference),
    y = difference
  )
) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    color = "grey50"
  ) +
  geom_col(
    fill = "#002D72",
    width = 0.7
  ) +
  coord_flip() +
  labs(
    title = "2026 Mets: September hit rates versus model predictions",
    subtitle = "Difference in percentage points; minimum 20 batted balls",
    x = NULL,
    y = "Actual minus predicted hit rate (percentage points)"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  "figures/mets_september.png",
  p2,
  width = 9,
  height = 6,
  dpi = 220
)

# Graph 3: Calibration of predicted hit probabilities.

p3_data <- calibration |>
  filter(
    model %in% c(
      "EV only (nonlinear degree 3)",
      "EV + angle (nonlinear degree 3)"
    )
  ) |>
  mutate(
    model = recode(
      model,
      "EV only (nonlinear degree 3)" = "EV only",
      "EV + angle (nonlinear degree 3)" = "EV + launch angle"
    )
  )

p3 <- ggplot(
  p3_data,
  aes(
    x = mean_prediction,
    y = observed_hit_rate,
    color = model,
    size = n
  )
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = "grey50"
  ) +
  geom_point(alpha = 0.85) +
  scale_size_area(max_size = 9) +
  coord_equal(
    xlim = c(0, 1),
    ylim = c(0, 1)
  ) +
  labs(
    title = "September 2026 probability calibration",
    subtitle = "Each point represents a predicted probability range",
    x = "Average predicted hit probability",
    y = "Observed hit rate",
    color = "Model",
    size = "Batted balls"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  "figures/calibration.png",
  p3,
  width = 8,
  height = 6,
  dpi = 220
)

# Graph 4: Contact quality heatmap.

p4_data <- d |>
  filter(
    between(launch_speed, 60, 119.999),
    between(launch_angle, -30, 59.999)
  ) |>
  mutate(
    ev_bin = floor(launch_speed / 5) * 5,
    angle_bin = floor(launch_angle / 10) * 10
  ) |>
  group_by(ev_bin, angle_bin) |>
  summarise(
    n = n(),
    hit_rate = mean(hit),
    .groups = "drop"
  ) |>
  filter(n >= 30)

p4 <- ggplot(
  p4_data,
  aes(
    x = ev_bin,
    y = angle_bin,
    fill = hit_rate
  )
) +
  geom_tile(width = 5, height = 10) +
  scale_fill_viridis_c(
    limits = c(0, 1),
    labels = scales::percent
  ) +
  labs(
    title = "2026 hit rate by exit velocity and launch angle",
    subtitle = "Each cell contains at least 30 tracked batted balls",
    x = "Exit velocity (mph)",
    y = "Launch angle (degrees)",
    fill = "Hit rate"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  "figures/contact_heatmap.png",
  p4,
  width = 9,
  height = 5.5,
  dpi = 220
)

# Estimate uncertainty in the September Brier score improvement.
# Resample entire games rather than individual batted balls.

set.seed(2026)

by_game <- test |>
  mutate(
    base_sq = (hit - prediction_2)^2,
    both_sq = (hit - prediction_4)^2
  ) |>
  group_by(game_pk) |>
  summarise(
    n = n(),
    base_sum = sum(base_sq),
    both_sum = sum(both_sq),
    .groups = "drop"
  )

ng <- nrow(by_game)

boot <- replicate(2000, {
  
  ix <- sample.int(
    ng,
    size = ng,
    replace = TRUE
  )
  
  sum(
    by_game$base_sum[ix] -
      by_game$both_sum[ix]
  ) / sum(by_game$n[ix])
})

observed <- mean(
  (test$hit - test$prediction_2)^2
) - mean(
  (test$hit - test$prediction_4)^2
)

ci <- quantile(
  boot,
  c(0.025, 0.975)
)

uncertainty <- data.frame(
  n_games = ng,
  brier_improvement = observed,
  ci_low = unname(ci[1]),
  ci_high = unname(ci[2]),
  bootstrap_reps = length(boot)
)

write_csv(
  uncertainty,
  "outputs/september_game_bootstrap_R.csv"
)

print(uncertainty)

cat(
  "\nAnalysis complete!\n",
  "Results saved in outputs/.\n",
  "Graphs saved in figures/.\n"
)

# Mets First-Half and Second-Half Analysis
# Compare Mets hitting before and after the All-Star break.

mets_halves <- d |>
  filter(batting_team == "NYM") |>
  mutate(
    half = case_when(
      date <= as.Date("2026-07-12") ~ "First half",
      date >= as.Date("2026-07-16") ~ "Second half",
      TRUE ~ NA_character_
    )
  ) |>
  filter(!is.na(half))

# Team-wide contact quality and hit rates.

mets_half_summary <- mets_halves |>
  group_by(half) |>
  summarise(
    batted_balls = n(),
    hits = sum(hit),
    hit_rate = mean(hit),
    avg_ev = mean(launch_speed),
    median_ev = median(launch_speed),
    avg_angle = mean(launch_angle),
    hard_hit_rate = mean(launch_speed >= 95),
    games = n_distinct(game_pk),
    .groups = "drop"
  )

write_csv(
  mets_half_summary,
  "outputs/mets_half_summary.csv"
)

cat("\nMets first-half vs. second-half contact:\n")
print(mets_half_summary, width = Inf)

# Compare individual hitters across both halves.

mets_player_halves <- mets_halves |>
  group_by(batter, half) |>
  summarise(
    n = n(),
    hits = sum(hit),
    hit_rate = mean(hit),
    avg_ev = mean(launch_speed),
    avg_angle = mean(launch_angle),
    hard_hit_rate = mean(launch_speed >= 95),
    .groups = "drop"
  ) |>
  add_names() |>
  arrange(player, half)

write_csv(
  mets_player_halves,
  "outputs/mets_player_halves.csv"
)

# For the direct comparison, require at least
# 40 tracked batted balls in each half.

mets_player_changes <- mets_player_halves |>
  select(
    batter, player, half, n,
    hit_rate, avg_ev, avg_angle, hard_hit_rate
  ) |>
  tidyr::pivot_wider(
    names_from = half,
    values_from = c(
      n, hit_rate, avg_ev,
      avg_angle, hard_hit_rate
    ),
    names_sep = "_"
  ) |>
  filter(
    !is.na(`n_First half`),
    !is.na(`n_Second half`),
    `n_First half` >= 40,
    `n_Second half` >= 40
  ) |>
  mutate(
    hit_rate_change_pp = 100 * (
      `hit_rate_Second half` -
        `hit_rate_First half`
    ),
    ev_change = `avg_ev_Second half` -
      `avg_ev_First half`,
    hard_hit_change_pp = 100 * (
      `hard_hit_rate_Second half` -
        `hard_hit_rate_First half`
    )
  ) |>
  arrange(desc(hit_rate_change_pp))

write_csv(
  mets_player_changes,
  "outputs/mets_player_changes.csv"
)

cat("\nMets hitters with 40+ batted balls in each half:\n")

print(
  mets_player_changes |>
    select(
      player,
      `n_First half`,
      `n_Second half`,
      hit_rate_change_pp,
      ev_change,
      hard_hit_change_pp
    ),
  width = Inf
)

# July-September Out-of-Sample Mets Predictions
# Train the nonlinear model on preceding months
# and save predictions for each held-out month.

mets_predictions <- list()

for (test_month in 7:9) {
  
  training <- d |>
    filter(
      as.integer(format(date, "%m")) < test_month
    )
  
  testing <- d |>
    filter(
      as.integer(format(date, "%m")) == test_month
    )
  
  fitted_model <- fit_model(
    training,
    include_angle = TRUE,
    degree = 3
  )
  
  testing$predicted_hit <- predict_model(
    fitted_model,
    testing
  )
  
  mets_predictions[[length(mets_predictions) + 1]] <-
    testing |>
    filter(batting_team == "NYM") |>
    mutate(
      test_month = month.name[test_month]
    )
}

mets_oos <- bind_rows(mets_predictions)

# Summarize the pooled out-of-sample predictions.

mets_oos_summary <- mets_oos |>
  group_by(batter) |>
  summarise(
    n = n(),
    actual_hits = sum(hit),
    expected_hits = sum(predicted_hit),
    actual_rate = mean(hit),
    expected_rate = mean(predicted_hit),
    .groups = "drop"
  ) |>
  mutate(
    difference_hits = actual_hits - expected_hits,
    difference_pp = 100 * (
      actual_rate - expected_rate
    )
  ) |>
  add_names() |>
  arrange(desc(n))

write_csv(
  mets_oos_summary,
  "outputs/mets_july_september_predictions.csv"
)

cat("\nMets July-September out-of-sample predictions:\n")

print(
  mets_oos_summary |>
    filter(n >= 40) |>
    select(
      player, n, actual_hits,
      expected_hits, difference_hits,
      difference_pp
    ),
  width = Inf
)

cat("\nTotal Mets batted balls in July-September:",
    nrow(mets_oos), "\n")

cat("Actual hits:", sum(mets_oos$hit), "\n")

cat("Expected hits:",
    round(sum(mets_oos$predicted_hit), 1), "\n")

cat("\nExtended Mets analysis complete!\n")

# Compare changes in exit velocity and hit rate
# for Mets hitters between the two halves.

p5 <- ggplot(
  mets_player_changes,
  aes(
    x = ev_change,
    y = hit_rate_change_pp,
    label = player
  )
) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    color = "grey60"
  ) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey60"
  ) +
  geom_point(
    color = "#002D72",
    size = 3
  ) +
  geom_text(
    hjust = -0.1,
    vjust = -0.5,
    check_overlap = TRUE,
    size = 3
  ) +
  scale_x_continuous(
    expand = expansion(mult = c(0.1, 0.3))
  ) +
  labs(
    title = "2026 Mets: Changes in contact and hit rate",
    subtitle = "First half versus second half; minimum 40 batted balls in each half",
    x = "Change in average exit velocity (mph)",
    y = "Change in batted-ball hit rate (percentage points)"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  "figures/mets_half_comparison.png",
  p5,
  width = 11,
  height = 7,
  dpi = 220
)