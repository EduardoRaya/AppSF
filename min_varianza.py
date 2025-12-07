import numpy as np
import pandas as pd
import streamlit as st
import scipy.optimize as op
import sf_library as sfl      # Nuestra libreria

# =====================================================================
# Lista de activos a incluir en el portafolio
# =====================================================================
tickers = [
    'XLK','XLF','XLV','XLP','XLY','XLE','XLI','XLC','XLB','XLU','XLRE'
]

# =====================================================================
# Cargar retornos diarios de cada activo y unirlos en un solo DataFrame
# =====================================================================
all_returns = []   # Lista donde se guardará un DataFrame por activo

for ticker in tickers:
    t = sfl.daily_return(ticker, data_dir="./MarketData")      # Cargar retornos
    t = t[['date', 'return']].rename(columns={'return': ticker})  # Renombramos la columna return por el ticker
    all_returns.append(t)   # Lo guardamos en la lista

# Unimos todos los activos por la columna "date"
df = all_returns[0]
for t in all_returns[1:]:
    df = pd.merge(df, t, on='date', how='inner')   # Solo fechas compartidas

# Limpiamos y ordenamos
df = df.dropna().sort_values('date').reset_index(drop=True)

# Filtramos datos anteriores a 2019 (ventana histórica)
df = df[df['date']<'2019-01-01']

# =====================================================================
# Construcción de matriz de retornos y parámetros de Markowitz
# =====================================================================
mtx = df.drop(columns='date')    # Matriz solo de retornos
returns = mtx

# Retorno promedio anual de cada activo
mean_returns = returns.mean() * 252   # Multiplicado por días de mercado

# Matriz de covarianza anualizada
cov_matrix = returns.cov() * 252

n = len(tickers)   # Número de activos

# =====================================================================
# Función rendimiento y riesgo del portafolio
# =====================================================================
def portfolio_performance(weights):
    """
    Calcula retorno y volatilidad del portafolio.
    Fórmulas:
        rp = w^T μ
        σp = sqrt(w^T Σ w)
    """
    ret = np.dot(weights, mean_returns)                # w^T μ
    vol = np.sqrt(weights @ cov_matrix @ weights.T)    # sqrt(w^T Σ w)
    return ret, vol

# =====================================================================
# OPTIMIZACIÓN 1: Portafolio de mínima volatilidad
# =====================================================================
def minimize_volatility():
    x0 = np.ones(n)/n      # Condición inicial: pesos iguales
    bounds = tuple((0,1) for _ in range(n))  # Pesos entre 0 y 1 (no short)
    constraints = ({'type':'eq','fun':lambda w: np.sum(w)-1})  # Suma de pesos = 1

    # Minimizamos solo la volatilidad:
    # minimize( σp(w) )
    result = op.minimize(
        lambda w: portfolio_performance(w)[1],
        x0, constraints=constraints, bounds=bounds
    )
    return result.x, portfolio_performance(result.x)

# =====================================================================
# OPTIMIZACIÓN 2: Portafolio de máximo retorno
# =====================================================================
def maximize_return():
    x0 = np.ones(n)/n
    bounds = tuple((0,1) for _ in range(n))
    constraints = ({'type':'eq','fun':lambda w: np.sum(w)-1})

    # maximize(rp)  equivale a minimize(-rp)
    result = op.minimize(
        lambda w: -portfolio_performance(w)[0],
        x0, constraints=constraints, bounds=bounds
    )
    return result.x, portfolio_performance(result.x)

# =====================================================================
# OPTIMIZACIÓN 3: Portafolio de máximo Sharpe ratio
# =====================================================================
def maximize_sharpe(risk_free=0.0):
    """
    Sharpe ratio:
        S = (rp - rf) / σp
    maximize(S)  <--> minimize( -S )
    """
    x0 = np.ones(n)/n
    bounds = tuple((0,1) for _ in range(n))
    constraints = ({'type':'eq','fun':lambda w: np.sum(w)-1})

    def neg_sharpe(w):
        r, vol = portfolio_performance(w)
        return -(r - risk_free) / vol    # Negativo porque minimize() debe maximizar el Sharpe

    result = op.minimize(
        neg_sharpe, x0, constraints=constraints, bounds=bounds
    )
    return result.x, portfolio_performance(result.x)
# =====================================================================
# MODELO CLASICO 
# =====================================================================

def min_variance_given_return(target_return):

    # Punto inicial: pesos iguales
    x0 = np.ones(n) / n

    # Restricciones: suma de pesos = 1  y retorno objetivo
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},                     # suma = 1
        {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}  # retorno deseado
    )

    # No permitir posiciones cortas (0 a 1)
    bounds = tuple((0, 1) for _ in range(n))

    # Función objetivo: varianza
    def variance(w):
        return w @ cov_matrix @ w.T

    # Optimización
    result = op.minimize(
        variance,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # Retornar pesos + rendimiento y riesgo resultante
    w = result.x
    ret, vol = portfolio_performance(w)

    return w, ret, vol


# =====================================================================
# EJECUCIÓN DE LOS MODELOS
# =====================================================================

# Portafolio de mínima varianza
w_min, (ret_min, vol_min) = minimize_volatility()

# Portafolio de máximo retorno
w_ret, (ret_mx, vol_mx) = maximize_return()

# Portafolio de máximo Sharpe
w_sharpe, (ret_s, vol_s) = maximize_sharpe()

target = 0.30   # 12% retorno objetivo
w_opt, r_opt, vol_opt = min_variance_given_return(target)


# =====================================================================
# RESULTADOS
# =====================================================================

def print_portfolio_st(name, tickers, weights, ret, vol):
    st.subheader(name)

    df = pd.DataFrame({
        "Ticker": tickers,
        "Peso asignado": [f"{w:.4f}" for w in weights]
    })

    st.table(df)

    st.write(f"**Retorno esperado anual:** {ret:.4f}")
    st.write(f"**Volatilidad anual:** {vol:.4f}")


# Imprimir resultados con tickers
# print_portfolio_st("Portafolio Mínima Varianza", tickers, w_min, ret_min, vol_min)
# print_portfolio_st("Portafolio Máximo Retorno", tickers, w_ret, ret_mx, vol_mx)
# print_portfolio_st("Portafolio Máximo Sharpe", tickers, w_sharpe, ret_s, vol_s)
# print_portfolio_st("Portafolio Máximo Sharpe 12%", tickers, w_opt, r_opt, vol_opt)

