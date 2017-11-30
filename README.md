[![Build
Status](https://travis-ci.org/FabianSchuetze/disagreement_changing_beliefs_and_volatility.svg?branch=master)](https://travis-ci.org/FabianSchuetze/disagreement_changing_beliefs_and_volatility)

# Disagreement, Changing Beliefs and Stock Market Volatility

The Github folder contains the data used in the paper [Disagreement, Changing
Beliefs and Stock Market
Volatility](https://www.dropbox.com/s/bl4slvzkloi911s/jmp.pdf?dl=0&m=). The
data can be used for two things.  First, to describe the relationship between
disagreement, stock market volatlity and excess returns. Second, one can study
whether forecasters remain consistently more optimistic (pessimistic) than
their peers over time.

### Example Usage

#### The Disagreement_Data.py File
To see how disagreement among forecasters is related to future volatility,
controlling for time-varying risk-aversion, one can do the following:

```
print('EXAMPLE 1:')
from Disagreement_Data import disagreement_data
import os
cwd = os.getcwd()
wd = os.path.join(cwd, '../Data/')
disagreement = disagreement_data(wd)
survey, summary = disagreement.data(method='arma')
independent = ['disagreement', 'risk']
regression_output = disagreement.regression(summary, 'vol', independent)
regression_output.summary()
print(regression_output.summary())
```
The string ``wd`` specifies the path to the folder with the data. The code
above loads first the ``pd.DataFrame`` ``summary`` containing all relevant
variables. Several parameters for the function ``data`` can be used each
specify how raw volatility shall be filtered. The methods are described in the
docstring of ``data``. To compute how ``disagreement`` relates to
``volatility`` in financial markets one can use several control variables. In
the example, a measure of time-varying risk aversion ``risk`` is used. The
results of regressing ``vol`` on ``risk`` and ``disagreement`` are stored in
``regression_output``.

#### The Survey_Data.py File
How forecaters switch beliefs can be studied with the file ``Survey_Data.py``.

```
print('EXAMPLE 2:')
from Survey_Data import survey_data
import os
cwd = os.getcwd()
wd = os.path.join(cwd, '../Data/')
survey = survey_data(wd)
transitions = survey.transition_probabilities(4)
print(transitions)
```
The function ``survey.transistion_probabilities(4)`` calculates the probability
that a forecaster switches beliefs at least once over 4 periods.

### Data Sources
