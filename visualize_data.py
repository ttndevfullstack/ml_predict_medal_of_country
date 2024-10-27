# ==============================================================================
# Visualize data through charts
# ==============================================================================

# Correlation matrix
numeric_teams = teams.select_dtypes(
    include="number"
)  # Filter columns is numeric data
numeric_teams.corr()["medals"]

# Create a linear regression
sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)
sns.lmplot(x="age", y="medals", data=teams, fit_reg=True, ci=None)

# How many teams have won a certain number of medals.
teams.plot.hist(y="medals")