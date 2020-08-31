import pandas as pd
import streamlit as st
import yfinance
import ta
from ta import add_all_ta_features
from ta.utils import dropna

@st.cache
def load_data():
    components = pd.read_html('https://finance.yahoo.com/cryptocurrencies/')[0]
    return components.drop('1 Day Chart', axis=1).set_index('Symbol')


@st.cache
def load_quotes(asset):
    return yfinance.download(asset)

def main():
    components = load_data()
    title = st.empty()
    st.sidebar.title("Options")

    def label(symbol):
        a = components.loc[symbol]
        return symbol + '-' + a.Name

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
    rsi = data3[-section:]['momentum_rsi'].to_frame('momentum_rsi')

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
    st.line_chart(data2)

    st.subheader('Apply Technical Indicators')
    st.code("""
    data3 = data.copy()
    data3 = ta.add_all_ta_features(data3, "Open", "High", "Low", "Close", "Volume", fillna=True)
    """, language="python")
    st.dataframe(data3)

    st.subheader('Momentum Indicator: RSI')
    st.line_chart(rsi)

    if st.sidebar.checkbox('View statistic'):
        st.subheader('Stadistic')
        st.table(data2.describe())

    if st.sidebar.checkbox('View quotes'):
        st.subheader(f'{asset} historical data')
        st.write(data2)

    st.sidebar.title("About")
    st.sidebar.info('This app is a simple example of')

if __name__ == '__main__':
    main()
