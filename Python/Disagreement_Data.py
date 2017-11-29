r"""
filename: Disagreement_Data.py

Author: Fabian Schuetze

This file contains all ingredients for analyzing the relationship between
disagreement and financial variables, controlling for time-varying risk
aversion and business cycle Statistics.
"""
import sys
import pandas as pd
import numpy as np
import statsmodels.tsa.api as tsa
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
plt.close('all')

class disagreement_data(object):
    """
    Class to Generate all relevant statistics related to the survey data

    Parameters:
    -----------

    wd: string:
        The absolute path in which the data is stored.

    Examples:
    ---------
    >> SPF = SurveyData(LoadData = True)
    >> SPF.PearsonStandardErrors(lag = 1)
    """

    def __init__(self, wd):
        self.wd = wd

    def data(self, method):
        """
        This function obtains all the relevant data from harddisk.  The data is
        index at the time to which it refers to, not with the time the
        forecasts or estiamtes were actually made.

        returns:
        -------
        survey: pd.DataFrame(float):
            The forecasts form the Survey of Professional Forecasters

        summary: pd.DataFrame(float):
            Various series containing disagreement, financial variables,
            time-varying risk aversion and gdp growth and others.
        """
        ## == Return Data == ##
        # import pdb; pdb.set_trace()
        ( returns, riskless, vol, dividend, cons, recession, gdp, survey
        ) = self._load_data()
        summary = self._transform_returns(returns, riskless)

        ## == Divends == ##
        dividend = dividend.loc['1969-01-01':]
        dividend.set_index(summary['1969Q1':].index, inplace=True)
        summary['Growth'] = dividend['Growth']

        ## == Survey Data == ##
        survey, Forecast = self._transform_expectations(survey, summary)
        summary['disagreement'] = Forecast['std']
        summary['Forecast'] = Forecast['median']
        summary['AbsError'] = Forecast['AbsError']

        ## Volatility and Volume == ##
        vol = self._realized_vol(vol, summary)
        if method == 'gallant':
            requirement = [('vol', 1)]
            summary['vol'] = self._TransformGallant(vol, requirement)
        elif method == 'ar':
            order = (1, 0)
            summary['vol'] = self._transform_arma(vol, order)
        elif method == 'arma':
            order = (1, 1)
            summary['vol'] = self._transform_arma(vol, order)
        else:
            summary['vol'] = vol

        ## == Consumption for Time-Varying risk aversion == ## 
        cons.set_index(summary.index, inplace=True)
        summary.loc[:, 'cons'] = cons['A796RX0Q048SBEA']
        summary['risk'] = self._time_varying_risk(summary)

        ## == Recession and GDP == ##
        quarterly = [recession.index.year, recession.index.quarter]
        recession = recession.groupby(quarterly).mean()
        summary['recession'] = recession.set_index(summary.index)
        summary['gdp'] = gdp.set_index(summary.index)
        summary['gdp_vol'] = self._gdp_vol_estimate(summary)

        summary.drop(['P', 'D', 'CPI'], axis=1, inplace=True)

        summary = summary.loc['1969Q1':,:]
        return survey, summary

    def _load_data(self):
        """
        Loads all the needed data from the harddrive
        """
        path_r = self.wd + 'Shiller_Monthly_Data_Revised.csv'
        path_rf = self.wd + 'Riskless.csv'
        path_dividend = self.wd + 'dividend.csv'
        path_cons = self.wd + 'non_durables.csv'
        path_recession = self.wd + 'Recession.csv'
        path_gdp = self.wd + 'GDP.csv'
        path_vol = self.wd + 'vol.csv'
        path_survey = self.wd + 'SurveyPhilLongNextProfits.csv'
        returns = pd.read_csv(path_r, header=0, index_col=0, parse_dates=True)
        rf = pd.read_csv(path_rf, header=0, index_col=0, parse_dates=True)
        dividend = pd.read_csv(path_dividend, header=0, index_col=0)
        vol = pd.read_csv(path_vol, header=0, index_col=0, parse_dates=True)
        recession = pd.read_csv(path_recession, header=0, index_col=0, parse_dates=True)
        gdp = pd.read_csv(path_gdp, header=0, index_col=0, parse_dates=True)
        cons = pd.read_csv(path_cons, header=0, index_col=0, parse_dates=True)
        survey = pd.read_csv(path_survey, header=0)
        return (returns, rf, vol, dividend, cons, recession, gdp, survey)

    def _surplus_paras(self, cons):
        """
        Helper function to estimate the time-varying risk aversion. calcualtes
        the paramters in the same way as Cochrane does
        """
        phi = 0.94**(1/4)
        gamma = 2.
        sigma = np.log(cons['cons']/cons['cons'].shift(4)).std()/(4**0.5)
        g = np.log(cons['cons']/  cons['cons'].shift(1)).mean()
        bar_S = sigma*(gamma/(1-phi))**0.5
        bar_s = np.log(bar_S)
        S_MAX = bar_s + 0.5*(1-bar_S**2)
        return S_MAX, bar_s, bar_S, g, sigma, phi, gamma

    def _lambda_S(self, s_MAX, bar_s, bar_S, s):
        """
        Helper function to calcualtes lambda_s form Campbell and Cochrane,
        function (10) in their paper.
        """
        if s < s_MAX:
            lambda_s = np.sqrt(1 - 2*(s - bar_s))/bar_S - 1
        else:
            lambda_s = 0.
        return lambda_s

    def _time_varying_risk(self, summary):
        """
        Calcualtes the time-varing risk aversion coefficient from 
        Campell and Cochanre (1999). \eta in their formulation

        Returns:
        --------

        risk: pd.series(float)
            An esimate of time-varing risk aversion. The calculation follows
            Cochrane and Campell (1999) and Cochrane's Discount Rates computer
            code
        """
        cons = summary.loc['1969Q1':,['cons']].copy(deep=True)
        s_MAX, bar_s, bar_S, g, sigma, phi, gamma = self._surplus_paras(cons)
        cons['log_surplus'] = np.ones(len(cons)) * bar_s
        CONST =  ( 1- phi) * bar_s
        for i, _ in cons.loc[:'2016Q3', :].iterrows():
            s = cons.loc[i, 'log_surplus']
            lambda_s = self._lambda_S(s_MAX, bar_s, bar_S, s)
            diff = np.log(cons.loc[i+1, 'cons'] / cons.loc[i, 'cons']) - g
            s_next = CONST + phi * s + lambda_s * diff
            cons.set_value(i+1, 'log_surplus', s_next)
        risk = gamma / np.exp(cons['log_surplus'])
        return risk 

    def _transform_expectations(self, survey, summary):
        """
        Modifies the raw survey data to make sense of it.
        """
        # import pdb; pdb.set_trace()
        survey['Date'] = survey.YEAR.map(str) + 'Q' + survey.QUARTER.map(str)
        newIndex = pd.PeriodIndex(survey.Date, freq='Q').shift(1)
        survey.set_index(newIndex, inplace=True)
        survey['Growth'] = np.log( survey['Forecast'] / survey['Nowcast'])
        survey['AbsError'] = np.abs(survey['Growth'] - summary.loc['1969Q1':,'Growth'])
        survey['Error'] = survey['Growth'] - summary.loc['1969Q1':,'Growth']
        quarterly = [survey.index.year, survey.index.quarter]
        grouped = survey.groupby(quarterly)
        Forecast = grouped.agg({'Growth': ['median', 'std']})['Growth']
        Forecast['AbsError'] = grouped.agg({'AbsError': ['mean']})['AbsError']
        Forecast.set_index(survey.index.unique(), inplace=True)
        survey['RelError'] = survey['AbsError'] / Forecast['AbsError'] 
        survey.dropna(inplace=True)
        keep = ['Growth', 'ID', 'AbsError', 'RelError', 'Error']
        return survey[keep], Forecast

    def _realized_vol(self, SP, summary):
        """
        Generate Trade Volumen _TransformReturfrom the SP Dataset.

        Parameters:
        -----------

        SP: pandas.dataframe(float):
            Daily SP500 data containing, among others,trader volumen and closing
            prices

        summary: pandas.dataframe(float):
            The Return index used so far

        Return:
        -------

        vol: pandas.dataframe(float):
            Quartlery data involing total trade volume and quarterly volatility
        """
        quarterly = [SP.index.year, SP.index.quarter]
        SP['returns'] = np.log(SP['Close'] / SP['Close'].shift(1))
        fun = lambda x: (x - x.mean())**2
        SP['vol'] = SP['returns'].groupby(quarterly).transform(fun)
        vol = SP[['vol']].groupby(quarterly).sum()
        vol.set_index(summary.index, inplace=True)
        vol['vol'] = vol['vol']**0.5
        return np.log(vol)

    def _regressor_gallant(self, data):
        """
        Helper Function to modfiy for the estimation of underlying volatility a
        la Gallant et al.

        Returns:
        --------
        regressor: np.array(float)
            Contains dummy variables for the different quarters and a linear
            and quadratic time trend
        """
        regressor = np.zeros((len(data), 6))
        regressor[:, 0] = 1
        for i in [2, 3, 4]:
            regressor[:, i-1][data.index.quarter == i] = 1
        time_index = np.array((range(1, len(data) + 1))) # detrending
        regressor[:, 4] = time_index / len(data)
        regressor[:, 5] = (time_index / len(data))**2
        return regressor

    def _TransformGallant(self, variables, series):
        """
        Transforms the variables according to Gallant et el (1992)

        Parameters:
        -----------

        Variables: pd.dataframe(float):
            A dataframe containing the data to be transformed

        series:   list(tuples):
            A list with tuples containing the series name and a boolean for
            whether a time-trend should be removed.
            Example: series  = [('Volume', 1), ('vol', 0)] removes a time-trend
            from Volumne but none from vol

        summary:
        --------

        Variables: pd.dataframe(float):
            The same dataframe but with partly altered series
        """
        data = variables.copy()
        message = 'Need to specify if time-trend in variables should be removed'
        assert all(isinstance(i, tuple) for i in series), message
        regressor = self._regressor_gallant(data)
        # import pdb; pdb.set_trace()
        for name, trend in series:
            if trend == 0:
                res_mean = sm.OLS(data[name], regressor[:, :4], missing='drop').fit()
            elif trend == 1:
                res_mean = sm.OLS(data[name], regressor, missing='drop').fit()
            resid_square = np.log(res_mean.resid**2)
            res_var = sm.OLS(resid_square, regressor, missing='drop').fit()
            store = res_mean.resid/np.exp(res_var.fittedvalues/2)
            beta1 = data[name].std()/store.std()
            beta0 = data[name].mean() - beta1 * store.mean()
            transformed_variable = beta0 + beta1 * store
        return transformed_variable

    def _gdp_vol_estimate(self, variables):
        """
        This function estimates conditional volatilities of GDP 
        growth
        """
        raw_vol = tsa.ARMA(variables['gdp'], order=(1, 0)).fit(disp=False).resid
        abs_vol = np.abs(raw_vol)
        filtered_vol = tsa.ARMA(abs_vol, order=(1, 0)).fit(disp=False)
        return filtered_vol.fittedvalues

    def _transform_arma(self, variables, order):
        """
        uses an arma(p, 0, q) model to compute vol

        Paramters:
        -----------
        variables: pd.DataFrame(float)
            The (log) realized volatility.

        order: tuple(int, int)
            The p and q terms for the ARMA(p,q) process

        Returns:
        --------
        res.fittedvalues pd.series(float)
            The fitted values of the arma model
        """
        res = tsa.ARMA(variables['vol'], order=order).fit(disp=False)
        return res.fittedvalues

    def _transform_returns(self, returns, riskless):
        """
        Function to modify the raw returns data and riskless interest rate so
        that it can be used
        """
        returns = returns.copy(deep=True)
        riskless = riskless.copy(deep=True)
        returns.set_index(riskless.index, inplace=True)
        quarters = np.in1d(returns.index.month, np.array([3, 6, 9, 12]))
        returns = returns.loc[quarters]
        riskless = riskless.loc[quarters]
        riskless = riskless / 4
        returns['D'] = returns['D'] / 4
        inflation = returns['CPI'] / returns['CPI'].shift(1)
        returns['riskfree'] = (1 + riskless['TB3MS'] / 100) / inflation
        returns['riskfree'] = np.log(returns['riskfree'])
        returns_numerator = returns['P'].shift(-1) + returns['D']
        returns['return'] = returns_numerator / returns['P']
        returns['return'] = np.log(returns['return'])
        returns['DP'] = np.log(returns['D'] / returns['P'])
        returns['ExcessReturn'] = returns['return'] - returns['riskfree']
        new_index = pd.PeriodIndex(returns.index, freq='Q')
        returns.set_index(new_index, inplace=True)
        returns['muR'] = self._expected_returns(returns, 'DP')
        return returns

    def Statistics(self, Variables, Series):
        """
        Calculates the same univariate summary statistics as in Table II
        of Lettau and Ludivigson (2001). Takes the names of the variables in
        reutrns
        """
        #import pdb; pdb.set_trace()
        TableCorr = Variables[Series].corr()
        TableCorr.values[np.tril_indices_from(TableCorr)] = np.nan
        np.fill_diagonal(TableCorr.values, 1)
        TableCorr = TableCorr.round(decimals = 2)
        TableUni = pd.DataFrame(index=['Mean', 'Std', 'Autocorrelation'],
                             columns=Series)
        TableUni.loc['Mean'] = Variables[Series].mean()
        TableUni.loc['Std'] = Variables[Series].std()
        for i in Series:
            TableUni[i].loc['Autocorrelation'] = Variables[i].autocorr()
        return TableCorr, TableUni

    def _standardize(self, df, label):
        """
        standardizes a series with name ``label'' within the pd.DataFrame
        ``df''.
        """
        df = df.copy(deep=True)
        series = df.loc[:, label]
        avg = series.mean()
        stdv = series.std()
        series = (series - avg)/ stdv
        return series

    def regression(self, data, label, independent, HAC=True, maxlength=1,
                   standardize=True):
        """
        computes robust standard errors
        """
        data = data.copy(deep=True)
        formula = label + ' ~ '
        for name in independent:
            formula = formula + ' + ' + name
        if standardize:
            for name in [independent, label]:
                data.loc[:, name] = self._standardize(data, name)
        ols = sm.ols(formula=formula, data=data)
        if HAC:
            cov_type = 'HAC'
            cov_kwds = {'maxlags' : maxlength}
            regression = ols.fit(cov_type=cov_type, cov_kwds=cov_kwds)
        else:
            regression = ols.fit()
        return regression
    
    def _expected_returns(self, df, label):
        """
        calcualtes expected excess returns
        """
        label_lag = label + '_lag'
        df[label_lag] = df[label].shift(1)
        formula = 'ExcessReturn ~ ' + label_lag
        res = sm.ols(formula=formula, data=df).fit()
        return res.fittedvalues

    def plots(self, df, Series, ax, ylabel=''):
        """
        This function returns contains for the relevant plots

        Parameters:
        -----------

        df: pd.DataFrame(float):
            A pandas DataFrame containing the variables to plot

        ax: matplotlib.axes:
            A matplotlib axes object in which the data is supposed to be plot

        Series: list(string):
            A list with strings stating which variables are supposed to be
            printed

        returns:
        -------

        ax: matplotlib.axes:
            The modifies axes object
        """
        if not (type(Series) == list):
            print('The Series input must be a list')
            sys.exit()
        PlotSeries = df.to_timestamp()
        y = np.array(PlotSeries['Recession'])
        for name in Series:
            ax.plot(PlotSeries.index, PlotSeries[name], label=name)
        low, high = ax.get_ylim()[0], ax.get_ylim()[1]
        ax.fill_between(PlotSeries.index, low, high,
                        where=y > 0., facecolor='grey')
        ax.set_ylim(bottom=low, top=high)
        ax.set_ylabel(ylabel)
        ax.legend()
        return ax

if __name__ == '__main__':
    wd = '/home/fabian/Documents/Eigene Text/NeuralNetworks_Publication/Data/'
    store = disagreement_data(wd=wd)
    survey, summary = store.data(method = 'arma')
