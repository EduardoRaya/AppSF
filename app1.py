import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List

# --- Constants: universes and default weights provided by the user
REGIONS = {
    'SPLG': 0.7062,
    'EWC': 0.0323,
    'IEUR': 0.1176,
    'EEM': 0.0902,
    'EWJ': 0.0537,
}

SECTORS = {
    'XLC': 0.0999,
    'XLY': 0.1025,
    'XLP': 0.0482,
    'XLE': 0.0295,
    'XLF': 0.1307,
    'XLV': 0.0958,
    'XLI': 0.0809,
    'XLB': 0.0166,
    'XLRE': 0.0187,
    'XLK': 0.3535,
    'XLU': 0.0237,
}

# --- Helper functions

def normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    vals = np.array(list(d.values()), dtype=float)
    s = vals.sum()
    if s == 0:
        # avoid division by zero: distribute equally
        n = len(vals)
        return {k: 1.0 / n for k in d.keys()}
    normalized = vals / s
    return {k: float(v) for k, v in zip(d.keys(), normalized)}


def sidebar_weight_inputs(tickers: List[str], defaults: List[float]) -> Dict[str, float]:
    """Render input widgets in a form and return a dict of weights."""
    weights = {}
    with st.form(key='weights_form'):
        st.write('Introduce los pesos para cada activo (puedes usar decimales, ejemplo 0.25 o 25).')
        use_percent = st.checkbox('Voy a introducir los pesos en porcentaje (0-100).', value=True)
        cols = st.columns(2)
        for i, t in enumerate(tickers):
            default = defaults[i]
            # convert default to percent for the input if use_percent
            default_input = round(default * 100, 4) if use_percent else default
            c = cols[i % 2]
            with c:
                val = st.number_input(label=f'{t}', min_value=0.0, max_value=10000.0, value=float(default_input), format='%.6f')
                weights[t] = float(val) / 100.0 if use_percent else float(val)
        normalize = st.checkbox('Normalizar pesos para que sumen 1.0 (recomendado).', value=True)
        submitted = st.form_submit_button('Guardar pesos')
        if submitted:
            if normalize:
                weights = normalize_weights(weights)
            st.session_state['weights'] = weights
            st.success('Pesos guardados en la sesión.')
    # return last-saved weights if available, otherwise defaults normalized
    if 'weights' in st.session_state:
        return st.session_state['weights']
    return normalize_weights({t: defaults[i] for i, t in enumerate(tickers)})


# --- Streamlit UI layout (Part 1)
st.set_page_config(page_title='Generador de portafolios - Parte 1', layout='wide')
st.title('Construcción de portafolios (Parte 1)')
st.markdown('Esta primera parte te permite seleccionar el universo de inversión y fijar los pesos de cada activo. Más adelante calcularemos estadísticas y optimizaciones.')

# Universe selection
universe = st.radio('Elige un universo de inversión:', ('Regiones', 'Sectores'))

if universe == 'Regiones':
    tickers = list(REGIONS.keys())
    defaults = list(REGIONS.values())
else:
    tickers = list(SECTORS.keys())
    defaults = list(SECTORS.values())

st.subheader(f'Activo y pesos por defecto ({universe})')
cols = st.columns([2, 1])
with cols[0]:
    st.table(pd.DataFrame({'Ticker': tickers, 'Peso por defecto': [f'{v*100:.2f}%' for v in defaults]}))
with cols[1]:
    st.write('Opciones:')
    st.write('- Portafolio arbitrario: define pesos manualmente y elige métricas para análisis cuantitativo (mínima varianza, máximo Sharpe, Markowitz).')
    st.write('- Portafolio optimizado (Black-Litterman): igual que anterior pero usando Black-Litterman para incorporar views.')

st.markdown('---')

# Input weights form
weights = sidebar_weight_inputs(tickers, defaults)

# Quick summary
st.subheader('Resumen de pesos actuales')
w_df = pd.DataFrame({'Ticker': list(weights.keys()), 'Peso': [f'{v:.6f}' for v in weights.values()]})
st.dataframe(w_df)

# Portfolio type and options
st.subheader('Elige el tipo de portafolio')
portfolio_option = st.selectbox('Tipo de portafolio', ['Portafolio arbitrario', 'Portafolio optimizado Black-Litterman'])

# If Arbitrary, let user choose which quantitative analyses to run later
if portfolio_option == 'Portafolio arbitrario':
    st.write('Selecciona las métricas/optimizaciones que quieres que se apliquen más adelante:')
    run_min_var = st.checkbox('Mínima varianza', value=True)
    run_max_sharpe = st.checkbox('Máximo Sharpe', value=True)
    run_markowitz = st.checkbox('Markowitz (frontera eficiente)', value=True)
else:
    st.write('Para Black-Litterman, podrás introducir views más adelante y elegir qué optimizaciones aplicar.')
    run_min_var = st.checkbox('Mínima varianza (usar con BL)', value=True)
    run_max_sharpe = st.checkbox('Máximo Sharpe (usar con BL)', value=True)
    run_markowitz = st.checkbox('Markowitz (usar con BL)', value=False)

# Risk-free and historical horizon inputs
st.subheader('Parámetros de datos')
col1, col2, col3 = st.columns(3)
with col1:
    hist_years = st.number_input('Años históricos para estimación (ej. 3)', min_value=1, max_value=20, value=3)
with col2:
    freq = st.selectbox('Frecuencia de datos', ['monthly', 'weekly', 'daily'])
with col3:
    rf = st.number_input('Tasa libre de riesgo anual (en decimal, ej. 0.03)', min_value=0.0, value=0.02)

st.markdown('---')

# Control buttons
col_a, col_b = st.columns(2)
with col_a:
    cont = st.button('Continuar a la siguiente parte (descarga de datos y cálculo)')
with col_b:
    edit = st.button('Quiero cambiar algo en esta parte')

# Store selections in session state for later parts
st.session_state['universe'] = universe
st.session_state['tickers'] = tickers
st.session_state['weights'] = weights
st.session_state['portfolio_option'] = portfolio_option
st.session_state['run_min_var'] = run_min_var
st.session_state['run_max_sharpe'] = run_max_sharpe
st.session_state['run_markowitz'] = run_markowitz
st.session_state['hist_years'] = hist_years
st.session_state['freq'] = freq
st.session_state['rf'] = rf

if cont:
    st.info('Has pulsado continuar: en la siguiente parte implementaremos la descarga de precios (yfinance), cálculo de retornos, covarianzas y las funciones de optimización (mínima varianza, máximo Sharpe, Markowitz), y la versión Black-Litterman.')

if edit:
    st.warning('Perfecto — modifica lo que necesites arriba y pulsa "Guardar pesos" para actualizar la sesión. Cuando estés listo, pulsa "Continuar".')

# End of Part 1

# Note: Parte 2 incluirá:
# - Descarga de datos (yfinance) con manejo de errores y ajuste de precios.
# - Cálculo de retornos esperados y matriz de covarianza según la frecuencia seleccionada.
# - Implementación de optimizadores: mínima varianza, máximo Sharpe, frontera eficiente (Markowitz).
# - Implementación de Black-Litterman para el Portafolio optimizado.
# - Visualizaciones: tabla de resultados, frontera eficiente plot y display de pesos.

# Cuando quieras que continúe, dime si quieres que incluya:
# 1) Soporte para constraints (peso mínimo/máximo por activo),
# 2) Importar views desde un CSV para Black-Litterman, o
# 3) Exportar resultados a Excel/PDF.
