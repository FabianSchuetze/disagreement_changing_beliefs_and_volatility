r"""
filename: Survey_Data.py

Author: Fabian Schuetze

This file contains all ingredients for analyzing the forecasts from the
Survey of Professional Forecasters as done in my paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

class survey_data(object):
    """
    Class to Generate all relevant statistics related to the survey data

    Parameters:
    -----------

    wd: string:
        The absolute path in which the data is stored.
    """

    def __init__(self, wd):
        self.wd = wd
        self.survey = self._DataCSV()
        self.periods = self.survey.index.unique()

    def _DataCSV(self):
        """
        This function loads the raw survey data and calculcates growth
        forecasts for each forecaster.

        Returns:
        -------
        survey: pd.DataFrame(float):
            The forecasts form the Survey of Professional Forecasters
        """
        survey_path = self.wd + 'SurveyPhilLongNextProfits.csv'
        survey = pd.read_csv(survey_path, header=0)
        survey['Date'] = survey.YEAR.map(str) + 'Q' + survey.QUARTER.map(str)
        new_index = pd.PeriodIndex(survey.Date, freq='Q').shift(1)
        survey.set_index(new_index, inplace=True)
        survey.dropna(inplace=True)
        survey['Growth'] = np.log( survey['Forecast'] / survey['Nowcast'])
        return survey[['Growth', 'ID']]

    def _remainingID(self, time, horizon, IDs):
        """
        Selctes the index numers of forecasts that submit forecasters for all
        periods

        Parameters:
        -----------

        time: scalar(int):
            The current index out of self.periods to determine the actual date

        horizon: scalar(int):
            The number of periods for which pessimism/optimism is considered

        IDs: array(int):
            The IDs of the forecasters at the beginning of the forecast

        Returns:
        --------

        ID: array(int):
            See above.

        """
        i = 1
        while i <= horizon:
            future = self.survey.loc[[self.periods[time + i]]]
            IDs = np.intersect1d(IDs, future['ID'])
            i += 1
        return IDs

    def _constantOpinion(self, time, horizon,
                         IDs, optimism, seperator, current):
        """
        Chose forecasters who remain optimistic or pessimstic throught the sample
        period

        Parameters:
        ----------

        time: scalar(int):
            The current index out of self.periods to determine the actual date

        horizon: scalar(int):
            The number of periods for which pessimism/optimism is considered

        IDs: array(int):
            The IDs of the forecasters at the beginning of the forecast

        optimism: boolean:
            If True

        seperator: scalar(int):
            Half the number of forecasters who submitted an initial forecaster.
            Only half of the forecasters can remain optimistic or pessimistic,
            serves as cut-off value.

        current: array(int):
            The IDs of the forecasters who submit a forecast for time t and their
            forecasts

        Returns:
        --------

        ranking: np.array(int64)
            The IDs of the forecasters who remaind consistently optimistic
            (if optimism == True ) or pessimistic (optimism == False)

        """
        low, high = seperator*optimism, seperator*(1 + optimism)
        SortedCurrent = current[np.argsort(current[:, 0])][:, 1] # IDs, sorted by beliefs
        ranking = SortedCurrent[low:high] #only opt/pes IDs
        i = 1
        while i <= horizon:
            future = self.survey.loc[[self.periods[time + i]]]
            future = np.array(future[future['ID'].isin(IDs)]) #subset of IDs
            SortedFuture = future[np.argsort(future[:, 0])][:, 1]
            futureRanking = SortedFuture[low:high]
            ranking = np.intersect1d(ranking, futureRanking)
            i += 1
        return ranking

    def _select_forecasters_experience(self, df, experience):
        """
        chooses only the subset of expeienced forecaters
        """
        df_new = df.copy(deep=True)
        F = list(df_new.ID)
        F = [i for i in list(set(F)) if F.count(i) >= experience * 4]
        df_new = df_new[df_new['ID'].isin(F)]
        return df_new

    def _transition_probabilities(self, horizon=1, experience=0):
        """
        This function calculates the probability that a forecaster who is more
        optimistic than her peers for the forecast at time t is also more optimistic
        for the forecast at time t + 1, ... , t + horizon.

        Parameters:
        -----------

        horizon: scalar(int):
            The number of periods for which are forecaters is required to stay more
            optimistic than her peers

        experience: scalar(int):
            If experience > 0, only forecasters which submitted forecasts for more
            than `experience` years are selected for the sample.

        Returns:
        --------

        p: pd.DataFrame:
            The dataframe with probabilities of remaining optimistic, pessimistic,
            the number of forecasters for each time and the weight in columns.
        """
        survey = self.survey
        prng = np.random.RandomState(1)
        if experience:
            survey = self._select_forecasters_experience(survey, experience)
        time = survey.index.unique()
        T = len(self.periods)
        p = pd.DataFrame(index=time, columns=['pLL', 'pHH', 'obs'])
        # import pdb; pdb.set_trace()
        for idx in range(0, T - horizon):
            current = survey.loc[[time[idx]]]
            # chose forecasters who remain in the survey from t to t+horizon :
            IDs = self._remainingID(idx, horizon, current['ID'])
            N = len(IDs)
            if N < 2:
                p.loc[time[idx]]['pLL'] = np.NaN
                p.loc[time[idx]]['pHH'] = np.NaN
                p.loc[time[idx]]['obs'] = N
                continue
            seperator = N // 2 + prng.binomial(1, N / 2 - N // 2) # half of the #ID
            current = np.array(current[current['ID'].isin(IDs)])
            Pes = self._constantOpinion(idx, horizon, IDs, False, seperator, current)
            Opt = self._constantOpinion(idx, horizon, IDs, True, seperator, current)
            p.loc[time[idx]]['pLL'] = len(Pes) / seperator
            p.loc[time[idx]]['pHH'] = len(Opt) / seperator
            p.loc[time[idx]]['obs'] = N
        p['weight'] = p['obs'] /p['obs'].sum()
        return p

    def transition_probabilities(self, horizon, experience=0):
        """
        Claulcates the probaiblity that a currenty optimistic forecaster
        switches to a pessimistic forecaster at least one over the subsequent
        `horzion' number of periods.

        Parameters:
        ------------

        horizon: float
            The number of periods considered for switiching beliefs

        experience: scalar(int):
            If experience > 0, only forecasters which submitted forecasts for more
            than `experience` years are selected for the sample.

        Returns:
        --------

        figures: pd.DataFrame
            A DaFrame with the probability of swichting at least once for each
            horizon
        """
        index = [1, 2, 3, 4]
        columns = ['Opt', 'Pes', 'AR(1) Model']
        figures = pd.DataFrame(index=index, columns=columns)
        for h in range(1, horizon + 1):
            p = self._transition_probabilities(horizon=h, experience=experience)
            statistics = np.zeros((2, 2))
            statistics[1, 0] = (p['weight'] * p['pLL']).sum()
            statistics[1, 1] = 1 - statistics[1, 0]
            statistics[0, 1] = (p['weight'] * p['pHH']).sum()
            statistics[0, 0] = 1 - statistics[0, 1]
            figures.loc[h, 'Opt'] = statistics[0, 1]
            figures.loc[h, 'Pes'] = statistics[1, 0]
            figures.loc[h, 'AR(1) Model'] = 0.6 ** h
        figures['average'] = (figures.loc[:, 'Opt'] + figures.loc[:, 'Pes']) * 0.5
        return 1 - figures

if __name__ == '__main__':
    wd = '/home/fabian/Documents/Eigene Text/NeuralNetworks_Publication/Data/'
    store = survey_data(wd=wd)
    ## == Obtain Data about the Transition Probabilities ##
    transitions = store.transition_probabilities(4)
    fig, ax = plt.subplots()
    red = [0.894, 0.101, 0.109]
    ax.plot(transitions['average'], label='Switching Probability', color=red)
    ax.xaxis.get_label().set_style('italic')
    ax.yaxis.get_label().set_style('italic')
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))
    ax.legend()
    ax.grid(linestyle='dashed', linewidth=5)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Probability')
    ax.set_xticks([1, 2, 3, 4])
