
# When Does Hitting the Ball Harder Matter?
### Predicting Offensive Outcomes Using 2026 MLB Statcast Data
Leo Kennedy  
George Washington University | Data Science

## Project Overview
Using 124,100 batted balls with tracking data from the 2026 MLB season, I explored whether including launch angle helps predict offensive outcomes beyond exit velocity alone. I fit four logistic regression models and compared them using data from July, August, and September as out-of-sample data. I also applied the best-performing model to predict hits for New York Mets batters, comparing expected and observed hits as well as changes in contact quality between the first and second halves of the season.

## Key Findings
- Models that include exit velocity and launch angle in a nonlinear fashion outperform models that only use exit velocity.
- Including launch angle decreases the Brier score by around **15.6%** in September when compared to the best nonlinear model that doesn't use launch angle. There were similar improvements in July and August.
- The observed hit rate on batted balls increased from **30.4% to 32.9%** between the first and second halves of the season for the Mets despite little change in hard-hit rate and exit velocity.
- The number of hits from July-September for the Mets on batted balls was **645**, while my model's prediction was around **638.3 hits**.
- Statcast's xBA metric beats out my models, so there's room for improvement!

## Methods and Tools
**Programming Language:** R

**Packages:** `readr`, `dplyr`, `ggplot2`, `glmnet`, `tidyr`, `scales`, `tibble`

**Statistical Methods:** Logistic regression, ridge regularization, nonlinear terms and interactions, rolling out-of-sample validation, Brier score, log loss, ROC AUC, probability calibration, and game-level bootstrap resampling.

**Data Source:** [MLB Baseball Savant - Statcast Search](https://baseballsavant.mlb.com/statcast_search)

## Project Files
| File or Folder | Description |
|---|---|
| `report.html` | Complete research report in HTML format |
| `report.pdf` | Research report in PDF format |
| `report.qmd` | Source file for report written in Quarto |
| `mets_analysis.R` | R script for data wrangling, modeling, analysis, and plots |
| `statcast_data/` | CSVs of Statcast data, broken up by month |
| `figures/` | Plots created using R |
| `outputs/` | Tables and other outputs from analysis |
| `styles.css` | CSS file for custom formatting in HTML report |
| `report_files/` | Other files used to support report in HTML format |

## Reproducing the Analysis
Clone or download this repository and open the folder in RStudio. Make sure you have the necessary R packages installed, and set your working directory to the main project folder. Running `mets_analysis.R` will recreate the analysis, plots, and model comparisons in this project.

To recreate the Quarto report, use the following commands:

```bash
quarto render report.qmd --to html
quarto render report.qmd --to pdf
```

Note that LaTeX must be installed to create a PDF (TinyTeX works well).

## Limitations
The data analyzed in this project is for batted balls with tracking data, not all plate appearances. My models also don't take into account other factors that can affect hits, like the direction the ball was hit, park effects, and fielder positioning. All the data used to fit and validate my models comes from the 2026 MLB season, so additional analysis would need to be done to see how well my models would perform on data from other seasons.

See the full report for more information on my analysis, model comparison, findings, and limitations!
