import statsmodels.api as sm


class ARIMA:
    def __init__(self, order, seasonal_order, trend):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.__res = None

    def fit(self, endog, exog=None):
        model = self.__create_model(endog)
        self.__res = model.fit()

    def predict(self, endog, exog=None):
        model = self.__create_model(endog)
        res = model.filter(self.__res.params)
        pred = res.predict()
        return pred

    def fitted_values(self):
        return self.__res.fittedvalues

    def __create_model(self, endog):
        model = sm.tsa.ARIMA(endog=endog, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend)
        return model
