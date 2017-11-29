# Disagreement, Changing Beliefs and Stock Market Volatility

The Github folder contains the data used in the paper "Disagreement, Changing
Beliefs and Stock Market Volatility". The data can be used for two things.
First, to describe the relationship between disagreement, stock market
volatlity and excess returns. Second, one can study whether forecasters remain
consistently more optimistic (pessimistic) than their peers over time.

### Example Usage

#### The Disagreement_Data.py File
To see how disagreement among forecasters is related to future volatility,
controling for time-varying risk-aversion, one can do the following:

```
from Disagreement_Data import disagreement_data
wd = '/home/fabian/Documents/texts/Disagreement/Data/'
disagreement = disagreement_data(wd)
survey, summary = disagreement.data(method='arma')
independent = ['disagreement', 'risk']
regression_output = disagreement.regression(summary, 'vol', independent)
regression_output.summary()
```
The string ``wd`` specifices the path to the folder with the data.  The code
above loads the first ``summary`` data containing all relevant variables.
Several measures of volatility are available, I use here the predicted values
of realized volatility, based on an ARMA(1,1) regression. I use ``disagreement`
and `risk` as exogeneous variables and calculate regression estimates with
``disagreement.regression``.

#### The Survey_Data.py File
How forecaters switch beliefs can be studied with the file ``Survey_Data.py``.

```
from Survey_Data import survey_data
wd = '/home/fabian/Documents/texts/Disagreement/Data/'
survey = survey_data(wd)
transitions = survey.transition_probabilities(4)
```
The function ``survey.transistion_probabilities(4)``` calculates the
probabiliyty that a forecasters switches beliefs at least once over 4 periods.

### Data Sources
