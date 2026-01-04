import statsmodels.api as sm

def build_arima(df):
    model = sm.tsa.ARIMA(df["Close"], order=(5,1,2))
    model_fit = model.fit()
    return model_fit
