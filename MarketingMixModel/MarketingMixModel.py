# -*- coding: utf-8 -*-
"""
Marketing Mix Model 
"""

import numpy as np
import scipy
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

class MarketingMixModel():
    
    def __init__(self, start_date='2019-06-01', end_date='2020-05-31'
                     , covid_st='2020-03-15', covid_ed='2020-05-31'
                     , spark_df=None, csv_path=None, excel_path=None):
        # input suppose to be a panda data frame
        self.is_spark_df = False
        
        if spark_df is not None:
            self.is_spark_df = True
            self.data = spark_df.fillna(0).toPandas()
            self.data.index = self.data['date']
            
        # read in if csv, or read a data table
        if csv_path is not None:
            self.data = pd.read_csv(csv_path, index_col='date')
            self.data = self.data.fillna(0)
        
        if excel_path is not None:
            self.data = pd.read_excel(excel_path, index_col='date')
            self.data = self.data.fillna(0)

        self.data.index = pd.to_datetime(self.data.index)
        self.data['c'] = 1
        self.data['isCovid'] = 0
        
        self.data = self.data.sort_index()
        
        # columns includes all features in data set
        self.columns = self.data.columns
        
        self.endog = None 
        self.exog = None
        self.lag = None
        
        # Label contains all features in the model
        self.labels = None
        self.coef = None
        self.E = None
        self.Cov = None
        
        self.irf = None
        self.irf_upper = None 
        self.irf_lower = None
        
        self.fevd = None
        
        if (start_date is None) or (end_date is None):
            print("Please specify model time frame" )
            return 0
        
        else:
            self.st = start_date
            self.ed = end_date
            
        if (covid_st is None) and (covid_ed is None):
            self.is_covid = False
            self.has_covid_data = False
        
        else:
            self.is_covid = True
            if covid_st <= start_date and covid_ed >= end_date:
                self.has_covid_data = False
                print("If whole period is covid or not covid, then dummy is not needed")
                return 0
            else: 
                self.has_covid_data = True
                self.covid_st = covid_st
                self.covid_ed = covid_ed
                self.data.loc[self.covid_st: self.covid_ed, 'isCovid'] = 1
        
    def corr(self):
        return pd.DataFrame(self.data.corr(), index=self.labels, columns=self.labels)
        
    def granger(self, var_A, var_B, lag=4):
    
        """
        Granger Causality Testing
        
        Parameters:
        -----------
        var_A : string 
            variable name tested as the cause
            
        var_B : string,
            variable name tested as the results
            
        Returns
        -------
            Results shows the statistical testing of if var_A will cause the change of var_B
        """
        
        data = self.data[[var_B, var_A]]
        gc_res = grangercausalitytests(data, lag)
        
        return gc_res[lag]
        
    def get_model_data(self):
        # convert to Pandas DF.
        return pd.DataFrame(self.Z.T, index=self.data.loc[self.st : self.ed].index, columns=self.labels)        
        
    def get_coef(self):    
        # convert to Pandas DF.
        return pd.DataFrame(self.coef, index=self.endog, columns=self.labels)

    def get_cov(self):
        # convert to Pandas DF.
        return pd.DataFrame(self.Cov, index=self.endog, columns=self.endog)
    
    def get_E(self):  
        # convert to Pandas DF.
        return pd.DataFrame(self.E, index=self.endog, columns=self.data.loc[self.st : self.ed].index)            
    
    def get_IRF(self, period):
        # convert to Pandas DF.
        return pd.DataFrame(self.irf[period], index=self.endog, columns=self.endog)
    
    def get_IRF_CI(self, period=-1):
        # convert to Pandas DF.
        return (pd.DataFrame(self.irf_upper[period], index=self.endog, columns=self.endog), 
                pd.DataFrame(self.irf_lower[period], index=self.endog, columns=self.endog))

    def get_FEVD(self, period=-1):
        # convert to Pandas DF.
        return pd.DataFrame(self.fevd[period], index=self.endog, columns=self.endog)
        
    def get_std(self, var=None):
        
        if self.is_spark_df == True:
            if var is None:
                data = self.data[self.endog].astype('float64', errors = 'ignore')
            else:
                data = self.data[var].astype('float64', errors = 'ignore')
        else:
            if var is None:
                data = self.data[self.endog]
            else:
                data = self.data[var]
                
        return data.loc[self.st : self.ed].std(axis=0)
    
    def __endog__(self, endog, p):
        
        """
        Internal function trying to convert endog matrix into modeling form
        
        Parameters:
        -----------
        endog : list 
            All endog variable names
            k is the number of parameters
            
        p : int,
            Number of lags need to be included in VAR estimation
            p is the number of lags  
            
        Returns
        -------
        Y : ndarray T * (kp)
            A data frame includes All records and endogenous variable include p lags
        """
        
        if self.is_spark_df == True:
            Y = self.data[endog].astype('float64', errors = 'ignore')
        else:
            Y = self.data[endog]
            
        for i in endog:
            for j in range(p):
                Y["{0}_{1}".format(i, j+1)] = Y[i].shift(periods=j+1)
                
        Z = Y[["{0}_{1}".format(i, j+1) for i in endog for j in range(p)]]

        return Z.loc[self.st : self.ed].to_numpy()
    
    
    def __exog__(self, exog):
        
        """
        Internal function trying to convert exog matrix into modeling form
        
        Parameters:
        -----------
        exog : list 
            All exog variable names
            m is the number of parameters
            
        Returns
        -------
        Y : ndarray T * m
            A data frame includes All records and m exogenous variable 
        """
        
        if self.is_spark_df == True:
            X = self.data[exog].astype('float64', errors = 'ignore')
        else:
            X = self.data[exog]
            
        # Add dummy convertion here if new variables in the list
        return X.loc[self.st : self.ed].to_numpy()
        
    def __weekday__(self):
        
        """
        Internal function convert weekday into binary dummies
        w_1 represents sunday
            
        Returns
        -------
        Y : ndarray T * 6
            A data frame includes All records and 6 weekday dummies
        """
        
        X = self.data.copy()
        
        X['dates'] = pd.to_datetime(X.index)
        X['wkday'] = X.dates.dt.weekday
        
        X['w_1'] = np.where(X['wkday'] == 6, 1, 0)
        X['w_2'] = np.where(X['wkday'] == 0, 1, 0)
        X['w_3'] = np.where(X['wkday'] == 1, 1, 0)
        X['w_4'] = np.where(X['wkday'] == 2, 1, 0)
        X['w_5'] = np.where(X['wkday'] == 3, 1, 0)
        X['w_6'] = np.where(X['wkday'] == 4, 1, 0)
    
        return X[['w_1', 'w_2', 'w_3', 'w_4', 'w_5', 'w_6']].loc[self.st : self.ed].to_numpy()
        
    
    def __fitting__(self, X, endog, exog):
        
        """
        Internal function try to fit the VAR model by feeding data and variables
        
        Equation Formulation 
        Y = BZ + E
        E is Error Matrix, suppose to be K * T
        B is Coefficient Matrix, suppose to be K * (pK+d), to be estimated
        Z is input Matrix, should be (pK+d) * T, p is lag, K is input, d is dummy (which is m exogenous variables plus 6 weekday dummy if included)
        """
        
        
        # Vectorizing Y Matrix
        if self.is_spark_df == True:
            Y = self.data[endog].astype('float64', errors = 'ignore').loc[self.st : self.ed].to_numpy()
        else:
            Y = self.data[endog].loc[self.st : self.ed].to_numpy()
        
        y = Y.T.flatten('F').reshape(Y.size, -1)

        I = np.identity((len(endog)))
        B = np.kron(np.linalg.inv(X @ X.T) @ X, I) @ y

        B = B.reshape(X.shape[0], -1).T
        E = Y.T - B @ X
        Cov = E @ E.T / (X.shape[1]-X.shape[0])

        return B, E, Cov
        
    def fit(self, endog=None, exog=['c'], lag=2, week_day=True, is_covid=True):
    
        """
        A process to fit the VAR model by inputting exdogeous variables,
        exogenous variable, also, lag should be specified
    
        Parameters
        ----------
        endog : list 
            All endog variable names
            k is the number of parameters
            
        exog : list
            Dummy variables included in VAR but not exodg
            n is the number of parameters
            
        lag : int,
            Number of lags need to be included in VAR estimation
            p is the number of lags    
            
        week_day: Boolean
            A parameter to decide if we want to include week of the day into model
            contribute to 6 parameters for each equation if included
            
        is_covid: Boolean
            A boolean indicated which allows modeling with or without covid dummy
            
        Notes
        -----
        VAR(p) model,
        
        if k variables, n dummy and p lags,
        
        total number of coefficients to estimate - k * (k * p + n + 6)
        6 will be included if we decide include week_day or not
        
        Returns
        -------
        coef : ndarray (k * (k * p + n)),
            All coefficients for VAR(p)
        
        E: ndarray (k * T),
            Error Matrics
            
        Cov: ndarray (k * k),
            Covariance Matrics of Prediction Error
        """
        
        self.is_covid = is_covid
        
        if exog is None:
            self.constant = False
        elif 'c' in exog:
            self.constant = True
        else:
            self.constant = False
            
        if week_day == True:
            self.week_day = True
        else:
            self.week_day = False
            
        # Update object
        self.endog = endog
        self.exog = exog 
        self.lag = lag
        
        # Extracting endogenous variables from the data
        Y = self.__endog__(endog, lag)

        if self.has_covid_data == True and self.is_covid == True:
            exog = exog + ['isCovid']
            
        # Extracting exogenous variables from the data
        if exog is not None:
            X = self.__exog__(exog)
        
        if week_day == True:
            wk = self.__weekday__()
            
        # Combine all the data sources available to build fitting sets.
        if exog is not None:
            if week_day == True:
                # self.labels = Y.columns + X.columns + wk.columns
                self.labels = ["{0}_{1}".format(i, j+1) for i in endog for j in range(lag)] + exog + ['w_1', 'w_2', 'w_3', 'w_4', 'w_5', 'w_6']
                Z = np.hstack((Y, X, wk)).T

            else:
                # self.labels = Y.columns + X.columns
                self.labels = ["{0}_{1}".format(i, j+1) for i in endog for j in range(lag)] + exog
                Z = np.hstack((Y, X)).T
                
        else:
            if week_day == True:
                # self.labels = Y.columns + wk.columns
                self.labels = ["{0}_{1}".format(i, j+1) for i in endog for j in range(lag)] + ['w_1', 'w_2', 'w_3', 'w_4', 'w_5', 'w_6']
                Z = np.hstack((Y, wk)).T

            else:
                # self.labels = Y.columns
                self.labels = ["{0}_{1}".format(i, j+1) for i in endog for j in range(lag)]
                Z = Y.T
        
        self.Z = Z
        # Fitting Model, Model Estimation
        self.coef, self.E, self.Cov = self.__fitting__(Z, endog, exog)

        print("Model Estimation is Done, create instance for coefficient, Error and Covariance For Error")
        
        
    def ma_rep(self, coefs, maxn=20):
    
        """
        MA(\infty) representation of VAR(p) process
    
        Parameters
        ----------
        coefs : ndarray (p x k x k)
        maxn : int
            Number of MA matrices to compute
    
        Notes 
        -----
        VAR(p) process as
    
        .. math:: y_t = A_1 y_{t-1} + \ldots + A_p y_{t-p} + u_t
    
        can be equivalently represented as
    
        .. math:: y_t = \mu + \sum_{i=0}^\infty \Phi_i u_{t-i}
    
        e.g. can recursively compute the \Phi_i matrices with \Phi_0 = I_k
    
        Returns
        -------
        phis : ndarray (maxn + 1 x k x k)
        """
        
        maxn = maxn - 1
        
        p, k, k = coefs.shape
        phis = np.zeros((maxn+1, k, k))
        phis[0] = np.eye(k)
    
        # recursively compute Phi matrices
        for i in range(1, maxn + 1):
            for j in range(1, i+1):
                if j > p:
                    break
    
                phis[i] += np.dot(phis[i-j], coefs[j-1])

        return phis
    
        
    def irf_std(self, n, t, p):
        
        # Using Bootstrap Simulation to create standard deviation band
        if self.is_spark_df == True:
            Y = self.data[self.endog].astype('float64', errors = 'ignore').loc[self.st : self.ed].to_numpy()
        else:
            Y = self.data[self.endog].loc[self.st : self.ed].to_numpy()
            
        data = self.get_model_data()
        coef = self.get_coef()
        
        # Shape T * m
        err = Y - data @ coef.T
                
        a_1, a_2, a_3 = self.irf.shape
        # build container and save all results 
        container = np.zeros((n, a_1, a_2, a_3))
        
        for loop in range(n):
            # sample with replacement
            index = np.random.choice(len(err), len(err), replace=True)
            samples = err.iloc[index].to_numpy()

            # construct new y
            Y = (data @ coef.T).to_numpy() + samples 

            # Refit model
            y = Y.T.flatten('F').reshape(Y.size, -1)
            X = self.Z

            I = np.identity((len(self.endog)))
            B = np.kron(np.linalg.inv(X @ X.T) @ X, I) @ y

            B = B.reshape(X.shape[0], -1).T
            E = Y.T - B @ X
            Cov = E @ E.T / (X.shape[1]-X.shape[0])

            # IRF Calculation
            coef_ = B[:, :len(self.endog)*self.lag]
            coef_list = []

            for i in range(self.lag):
                coef_list.append(coef_[:, i::self.lag])

            coef_stack = np.array(coef_list)
            impulse = self.ma_rep(coef_stack, maxn=t)
            
            if self.irf_method == 'C':
                # Chelesky Decomposition of Covariance Matrics
                P = np.linalg.cholesky(Cov)
                container[loop] = impulse @ P
            
            elif self.irf_method == 'G':
                container[loop] = impulse @ Cov / np.sqrt(np.diag(Cov))
                    
        std_upper = np.percentile(container, q=p, axis=0)
        std_lower = np.percentile(container, q=(100-p), axis=0)
        
        return std_upper, std_lower
        
    def irf_est(self, method='G', t=20, n=None, p=None):
        
        """
        # Once We have the coefficientS, we need to use it to generate the Generalized Impulse Response
        # According to Pesaran & Shin (1997), Equation <2.2>
        # 1st to represent VAR model using MA
        
        Parameters
        ----------
        coefs : ndarray (p x k x k)
            Coefficients estimated from VAR fitting process
        method: string
            'C' indicates Chelosky
            'G' indicates Generalized
        t : int 
            time period for the IRF 
        n : int
            numbers of iterations
        p : int
            percentile

        Returns
        -------
        irf : ndarray (maxn x k x k)
            Impulse Response Function,
        """
        
        coef_ = self.coef[:, :len(self.endog)*self.lag]
        coef_list = []
        
        for i in range(self.lag):
            coef_list.append(coef_[:, i::self.lag])
                
        coef_stack = np.array(coef_list)
        
        if method == 'G':
        
            self.irf_method = 'G'

            impulse = self.ma_rep(coef_stack, maxn=t)
            self.irf = impulse @ self.Cov / np.sqrt(np.diag(self.Cov))
        
        if method == 'C':
        
            self.irf_method = 'C'
            
            # Chelesky Decomposition of Covariance Matrics
            impulse = self.ma_rep(coef_stack, maxn=t)
            P = np.linalg.cholesky(self.Cov)
            self.irf = impulse @ P
        
        if n == None or p == None:
            return self.irf
        
        else:
            self.irf_upper, self.irf_lower = self.irf_std(n, t, p)
            return self.irf, self.irf_upper, self.irf_lower
        
        
    def fevd_est(self, t=20, adj_sig=False):
    
        """
        Once We have the  Impulse Response, we can generate FEVD
        According to Lanne & Nyberg 2016, Equation(9)
        
        Parameters
        ----------
        irf : ndarray (maxn x k x k)
            Impulse Response Function Results
        adj_sig : boolean
            Binary indicator, True means FEVD will only calculated based on significant part of IRF

        Returns
        -------
        fevd : ndarray (maxn x k x k)
            Forecast Error Variance Decomposition
        """
    
        self.irf_est(t=t, n=None, p=None)
        
        if adj_sig==True:
        
            if self.irf_lower is None:
                return "Please run IRF confidence interval at first"
                
            lower_bound = np.zeros(self.irf.shape)

            for j in range(self.irf.shape[1]):
                for k in range(self.irf.shape[2]):
                    for i in range(self.irf.shape[0]):

                        if (i < 2):
                            lower_bound[i, j, k] = self.irf[i, j, k]
                        else:
                            if self.irf_lower[i, j, k] < 0:
                                break
                            else:
                                lower_bound[i, j, k] = self.irf[i, j, k]

            self.irf = lower_bound
            
        irf_sq = np.power(self.irf, 2)
        
        irf_sq_cum = np.cumsum(irf_sq, axis=0)
        irf_sq_mar = np.sum(irf_sq_cum, axis=2)
        
        self.fevd = np.zeros(irf_sq_cum.shape)

        for i in range(t):
            for j in range(irf_sq_mar[i].shape[0]):
                self.fevd[i, j] = irf_sq_cum[i, j] / irf_sq_mar[i][j]
    
        return self.fevd
        
    def exogenous_impact(self, var='new_total_bkgs', exog=None, week_day=True):
        
        """
        Estimate Exogenous Impact of input exogenous variables and weekday dummy
        
        Parameters
        ----------
        var : string
            Impact in terms of ...
        exog : list
            A list of variables need to be estimated impact
        week_day : boolean 
            A boolean indicator determines if model will estimate week Dummy impact

        Returns
        -------
            A dictionary includes impact of all the listing variables
        """
        
        # Try to export a dictionary instead of List #
        impact = dict()
        
        # pulling model data
        Z = self.get_model_data()
        
        # Subset to most recent Month
        month1 = Z.index.max().month
        month2 = 12 if month1 - 1 == 0 else month1 - 1
        
        Z = Z[(Z.index.month == month1) | (Z.index.month == month2)]
        
        # Scoring Revenues without weekDay variables
        coef_ = self.get_coef()
        
        # Score Y, total Revenue
        weekdays = ['w_1', 'w_2', 'w_3', 'w_4', 'w_5', 'w_6']
        
        # Seasonality Impact
        Y = Z @ coef_.T
        impact['weekDay']  = (Z[weekdays] @ coef_.loc[var, weekdays].T).sum() / Y[var].sum()
            
        # Covid Impact
        if self.is_covid == True:
            covid = (Z['isCovid'] * coef_.loc[var, 'isCovid']).sum()
            impact['covid']  = covid / Y[var].sum()
            
        else:
            impact['covid']  = 0
        
        # Exogenous Impact
        if exog is not None:
            for exog_var in exog:
                
                exog_ = (Z[exog_var] * coef_.loc[var, exog_var]).sum()
                impact[exog_var] = exog_ / Y[var].sum()
                
        return impact