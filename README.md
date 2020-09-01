# Streamlit-Cryptocurrencies-Dashboard
About Streamlit Dashboard returning technical indicators for a given crypto (yfinance/ta).

##### Running: `streamlit run https://raw.githubusercontent.com/poltys/Streamlit-Cryptocurrencies-Dashboard/master/app.py`
![](https://github.com/poltys/Streamlit-Cryptocurrencies-Dashboard/blob/master/extra/streamlit-crypto-2020-08-31-17-08-89.gif)

#### To Do
- [X] Add option to download list with USD or EUR
  - [X] Defined eur-usd ticker conversion(https://github.com/poltys/Streamlit-Cryptocurrencies-Dashboard/blob/master/eur-usd-conversion/eur_usd_conversion.py):
    - however some ticker do not provide result
    - and first summary table is not updating
    - alternative source to be identified
- [X] Add user inputs(https://github.com/poltys/Streamlit-Dashboard-Ticker_Technical_Analysis)
  - [X] Buying Price
  - [ ] Add shapes(i.e. means)
  - [X] Create button to upload portfolio details and define global buying price / weighted av.
  - [ ] Portfolio Size
  - [X] T Price
  - [ ] Add Dynamic Indicators
    - [X] Momentum Indicators
    - [X] Volatility Indicators
  - [ ] Define df.style specific rules based on each technical indicators
  - [X] df.style apply to axis=1
- [ ] Add means
- [ ] Dynamically generate annotations
- [X] Add technical indicators
- [ ] Add already trained ML predictive model - google colab
- [ ] Deploy on Heroku
