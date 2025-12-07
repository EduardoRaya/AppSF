import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis

# ETFs de la estrategia 1: REGIONES DEL MUNDO
tickers_regiones = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]

# ETFs de la estrategia 2: SECTORES DE ESTADOS UNIDOS
tickers_sectores = ["XLC","XLY","XLP","XLE","XLF",
                    "XLV","XLI","XLB","XLRE","XLK","XLU"]

# Descargamos precios ajustados de los últimos años (aquí yo puse 4 pero podemos poner más o menos aunque menos
#entiendo que no es lo ideal porque no tendríamos tantos datos y más puede ya no reflejar el mercado actual)
data_regiones = yf.download(tickers_regiones, period="4y")["Close"]
data_sectores = yf.download(tickers_sectores, period="4y")["Close"]

#st.write("Precios por Regiones:")
#st.write(data_regiones.tail())
#st.write(data_regiones.tail())

#st.write("\nPrecios por Sectores:")
#st.write(data_regiones.tail())
#st.write(data_sectores.tail())

# Rendimientos diarios para REGIONES
retorno_regiones = data_regiones.pct_change().dropna()

# Rendimientos diarios para SECTORES
retorno_sectores = data_sectores.pct_change().dropna()

#st.write("Rendimientos REGIONES:")
#st.write(retorno_regiones.head())

#st.write("\nRendimientos SECTORES:")
#st.write(retorno_sectores.head())

#Pesos benchmark por REGIONES
pesos_regiones={"SPLG":0.7062,
                "EWC":0.0323,
                "IEUR":0.1176,
                "EEM":0.0902,
                "EWJ":0.0537}

pesos_sectores={"XLC":0.0999,
                "XLY":0.1025,
                "XLP":0.0482,
                "XLE":0.0295,
                "XLF":0.1307,
                "XLV":0.0958,
                "XLI": 0.0809,
                "XLB": 0.0166,
                "XLRE": 0.0187,
                "XLK": 0.3535,
                "XLU": 0.0237}


retorno_regiones = retorno_regiones[list(pesos_regiones.keys())]
retorno_sectores = retorno_sectores[list(pesos_sectores.keys())]

p_regiones = np.array([pesos_regiones[t] for t in retorno_regiones.columns])
p_sectores = np.array([pesos_sectores[t] for t in retorno_sectores.columns])

p_regiones = p_regiones / p_regiones.sum()
p_sectores = p_sectores / p_sectores.sum()

#print("Suma pesos REGIONES:", p_regiones.sum())
#print("Suma pesos SECTORES:", p_sectores.sum())

portafolio_regiones = (retorno_regiones * p_regiones).sum(axis=1)

portafolio_sectores = (retorno_sectores * p_sectores).sum(axis=1)

#st.write("Retorno del portafolio REGIONES (primeros días):")
#st.write(portafolio_regiones.head())

#st.write("\nRetorno portafolio SECTORES (primeros días):")
#st.write(portafolio_sectores.head())

def beta(port, benchmark):
    cov = np.cov(port, benchmark)[0,1]
    var = np.var(benchmark)
    return cov / var

def media(r):
    return r.mean()

def volatilidad(r):
    return r.std()

def sharpe(r, rf=0.0):
    excess = r - rf/252
    return np.sqrt(252) * excess.mean() / excess.std()

def sortino(r, rf=0.0):
    excess = r - rf/252
    downside = excess[excess < 0].std()
    return np.sqrt(252) * excess.mean() / downside

def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def var_95(r):
    return np.percentile(r, 5)

def cvar_95(r):
    v = var_95(r)
    return r[r <= v].mean()

def sesgo(r):
    return skew(r)

def curtosis(r):
    return kurtosis(r)

metrics_reg = {
    "Media diaria": media(portafolio_regiones),
    "Volatilidad diaria": volatilidad(portafolio_regiones),
    "Sharpe (5% rf)": sharpe(portafolio_regiones, rf=0.05),
    "Sortino (5% rf)": sortino(portafolio_regiones, rf=0.05),
    "Max Drawdown": max_drawdown(portafolio_regiones),
    "VaR 95%": var_95(portafolio_regiones),
    "CVaR 95%": cvar_95(portafolio_regiones),
    "Skew": sesgo(portafolio_regiones),
    "Kurtosis": curtosis(portafolio_regiones),
}

metrics_sec = {
    "Media diaria": media(portafolio_sectores),
    "Volatilidad diaria": volatilidad(portafolio_sectores),
    "Sharpe (5% rf)": sharpe(portafolio_sectores, rf=0.05),
    "Sortino (5% rf)": sortino(portafolio_sectores, rf=0.05),
    "Max Drawdown": max_drawdown(portafolio_sectores),
    "VaR 95%": var_95(portafolio_sectores),
    "CVaR 95%": cvar_95(portafolio_sectores),
    "Skew": sesgo(portafolio_sectores),
    "Kurtosis": curtosis(portafolio_sectores),
}


df_metrics = pd.DataFrame({
    "Regiones": metrics_reg,
    "Sectores": metrics_sec
})


st.title("Creación de portafolios")


def page_1():
    st.header("Regional", divider=True)

    option = st.selectbox(
    "Elige el tipo de portafolio:",
    ("Portafolio Arbitrario", "Portafolio Optimizado", "Portafolio Black Litterman"),
    index=None
    )

    if option == "Portafolio Arbitrario":
        st.subheader("Portafolio Arbitrario")
        st.write(df_metrics)
    elif option == "Portafolio Optimizado":
        st.subheader("Portafolio Optimizado")
        st.write(df_metrics)

def page_2():
    st.header("Regiones", divider=True)

    col = st.columns((1.5, 4.5, 2), gap='medium')

    with col[0]:
        st.markdown('#### Gains/Losses')

    with col[1]:
        st.markdown('#### Total Population')

    with col[2]:
        st.markdown('#### Top States')

pg = st.navigation([page_1, page_2])
pg.run()