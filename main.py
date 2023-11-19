import streamlit as st
from file_uploader import file_uploader_create
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

st.set_page_config(layout='wide', page_title='Linear regression')

st.markdown('<center><h1>Linear regression</h1></center>', unsafe_allow_html=True)
datas = file_uploader_create("Select dataset file")
datas = datas.select_dtypes(include='number')
try:

    x_cols = st.multiselect(
        'Select X(s):',
        datas.columns)
    y_col = st.selectbox(
        'Select Y:',
        datas.columns)

    st.write('Correlation matrix')
    st.write(datas.corr().loc[y_col])

    x = datas[x_cols]
    y = datas[y_col]
    lnrg = LinearRegression()
    model = lnrg.fit(x, y)
    y_p = lnrg.predict(x)
    coefficients = [x for x in model.coef_]
    coefficients.insert(0, model.intercept_)
    coefficients = pd.DataFrame(coefficients).T
    cols = ['b' + str(i) for i in range(0, len(model.coef_) + 1)]
    cols[0] = 'a'
    coefficients.columns = cols
    coefficients.set_index(np.array(['coefficients']), inplace=True)
    st.write(coefficients)
    coefficients = coefficients.iloc[0]

    datas = datas[list(x_cols) + list([y_col])]

    datas[y_col + '_predicted'] = y_p
    cols = st.columns([0.5, 0.5])
    with cols[0]:
        table_header = st.selectbox('Table head', [i for i in range(1, len(datas) + 1)])
    with cols[1]:
        st.write('Prediction accuracy score is:')
        st.write(str(model.score(x, y) * 100) + '%')

    st.markdown(datas.head(int(table_header)).style.hide(axis='index').to_html(), unsafe_allow_html=True)

except:
    pass
