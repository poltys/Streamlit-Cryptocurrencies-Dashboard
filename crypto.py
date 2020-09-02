import pandas as pd
import streamlit as st
import yfinance
import ta
from ta import add_all_ta_features
from ta.utils import dropna
# ml lib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow.keras.backend as K
import os


# file upload behavior to change
st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def load_data():
    components = pd.read_html('https://finance.yahoo.com/cryptocurrencies/')[0]
    return components.drop('1 Day Chart', axis=1).set_index('Symbol')

#laoding model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('BTC.h5')
    model.summary()
    return model


@st.cache
def load_quotes(asset):
    return yfinance.download(asset)

def main():
    components = load_data()
    title = st.empty()
    st.sidebar.title("Options")

    def label(symbol):
        a = components.loc[symbol]
        return symbol

    if st.sidebar.checkbox('View cryptocurrencies list'):
        st.dataframe(components[['Name',
                                 'Price (Intraday)',
                                 'Change',
                                 '% Change',
                                 'Market Cap']])

    st.sidebar.subheader('Select asset')
    asset = st.sidebar.selectbox('Click below to select a new asset',
                                 components.index.sort_values(), index=3,
                                 format_func=label)

    title.title(components.loc[asset].Name)
    if st.sidebar.checkbox('View cryptocurrencies info', True):
        st.table(components.loc[asset])
    data0 = load_quotes(asset)
    data = data0.copy().dropna()
    data.index.name = None

    section = st.sidebar.slider('Number of quotes', min_value=30,
                        max_value=min([2000, data.shape[0]]),
                        value=500,  step=10)

    data2 = data[-section:]['Adj Close'].to_frame('Adj Close')
    data3 = data.copy()
    data3 = ta.add_all_ta_features(data3, "Open", "High", "Low", "Close", "Volume", fillna=True)
    momentum = data3[['momentum_rsi', 'momentum_roc', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'momentum_ao', 'momentum_kama']]
    volatility = data3[['volatility_atr','volatility_bbm','volatility_bbh','volatility_bbl','volatility_bbw','volatility_bbp','volatility_bbhi','volatility_bbli','volatility_kcc','volatility_kch','volatility_kcl','volatility_kcw','volatility_kcp','volatility_kchi','volatility_kcli','volatility_dcl','volatility_dch',]]
    datapred = data[-section:]['Adj Close'].to_frame('Adj Close').rename(columns={'Adj Close':'adjclose'})

    sma = st.sidebar.checkbox('SMA')
    if sma:
        period= st.sidebar.slider('SMA period', min_value=5, max_value=500,
                             value=20,  step=1)
        data[f'SMA {period}'] = data['Adj Close'].rolling(period ).mean()
        data2[f'SMA {period}'] = data[f'SMA {period}'].reindex(data2.index)

    sma2 = st.sidebar.checkbox('SMA2')
    if sma2:
        period2= st.sidebar.slider('SMA2 period', min_value=5, max_value=500,
                             value=100,  step=1)
        data[f'SMA2 {period2}'] = data['Adj Close'].rolling(period2).mean()
        data2[f'SMA2 {period2}'] = data[f'SMA2 {period2}'].reindex(data2.index)

    st.subheader('Chart')
    if asset != 'BTC-USD':
        st.line_chart(data2)
    if asset == 'BTC-USD':
        model = load_model()
        #future_price = model.predict(model, datapred)
        st.line_chart(data2)

    if st.sidebar.checkbox('View momentum indicators'):
        st.subheader('Apply Technical Indicators')
        st.code("""
        data = ta.add_all_ta_features(data3, "Open", "High", "Low", "Close", "Volume", fillna=True)
        """, language="python")
        st.header(f'Momentum Indicators')
        transpose = momentum.iloc[[-5, -4, -3, -2, -1]].transpose()
        st.table(transpose.style.background_gradient(cmap='Blues', axis=1))
        for col in momentum.columns:
            st.subheader(f'Momentum Indicator: {col}')
            st.line_chart(data3[-section:][col].to_frame(col))

    if st.sidebar.checkbox('View volatility indicators'):
        st.subheader('Apply Technical Indicators')
        st.code("""
        data = ta.add_all_ta_features(data3, "Open", "High", "Low", "Close", "Volume", fillna=True)
        """, language="python")
        st.header(f'Volatility Indicators')
        transpose = volatility.iloc[[-5, -4, -3, -2, -1]].transpose()
        st.table(transpose.style.background_gradient(cmap='Blues', axis=1))
        for col in volatility.columns:
            st.subheader(f'Momentum Indicator: {col}')
            st.line_chart(data3[-section:][col].to_frame(col))

    if st.sidebar.checkbox('Personal portfolio analysis'):
        st.subheader(f'{asset} personal portfolio analysis')
        file_buffer = st.file_uploader("Choose a .csv or .xlxs file\n 2 columns are expected 'rate' and 'price'", type=['xlsx','csv'])
        if file_buffer is not None:
             file = pd.read_excel(file_buffer)
             file = pd.DataFrame(file)
             st.table(file.style.background_gradient(cmap='Blues'))
             weighted_rate = (file['price']*file['rate']).sum() / file['price'].sum()
             daily_price = data.Close.iloc[-1]
             perf = {'buying price': weighted_rate, 'current price': daily_price}
             performance = pd.DataFrame(perf, columns = ['buying price', 'current price'], index=[asset])
             st.table(performance.style.background_gradient(cmap='Blues', axis=1))

    if st.sidebar.checkbox('View statistic'):
        st.subheader('Statistic')
        st.table(data2.describe())

    if st.sidebar.checkbox('View quotes'):
        st.subheader(f'{asset} historical data')
        st.write(data2)


    st.sidebar.title("About")
    st.sidebar.info('This app aims to provide a comprehensive dashbaord to analyse cryptocurrency performance')

if __name__ == '__main__':
    main()
