import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd 
import seaborn as sns
from db_manager import Postgres
from datetime import datetime, timedelta

sns.set(rc={'figure.figsize':(21, 8)})

def load_data():
    db = Postgres()
    euribor = pd.read_sql(
        'SELECT eur_date, eur_12m, eur_year FROM orchard.euribor', 
        db.conn
        )
    df_eur = euribor.set_index('eur_date')
    df_eur.index = pd.to_datetime(df_eur.index)
    df_eur = df_eur.astype({'eur_year': np.int16})
    return df_eur

def plot_classic(df, col, year_filter=None):
    fig, ax = plt.subplots() 
    if year_filter:
        pass
    else:
        ax = df[col].plot(linewidth=0.5);
        ax.set_ylabel('Daily Euribor Rate');
    return fig 

def compare_years(df, start, end, col='eur_12m'):
    year = timedelta(365)
    two_years = timedelta(730)

    temp_weekly_mean = df[[col]].resample('W').mean()

    euribor_2019 = df.loc[start-year:end-year, col]
    euribor_2019.index = euribor_2019.index + pd.DateOffset(days=365)

    euribor_2018 = df.loc[start-two_years:end-two_years, col]
    euribor_2018.index = euribor_2018.index + pd.DateOffset(days=365*2)

    # Plot daily and weekly resampled time series together
    fig, ax = plt.subplots(figsize=(20,15))

    # Fisrt Plot 
    ax.plot(
        df.loc[start:end, col],
        marker='.', linestyle='-', linewidth=0.5, label='Daily 2020'
    )

    # Second Plot
    ax.plot(
        temp_weekly_mean.loc[start:end, col],
        marker='o', markersize=2, linestyle='-', label='Weekly Mean'
    )

    # euribor_2019
    ax.plot(
        euribor_2019,
        linestyle='-', linewidth=0.5, label='Daily 2019'
    )

    # euribor_2018
    ax.plot(
        euribor_2018,
      linestyle='-', linewidth=0.5, label='Daily 2018'
    )

    ax.set_ylabel('Euribor')
    ax.legend()
    return fig

df = load_data()



# Draw stuff
st.markdown("### Euribor record")
st.table(df.tail())
# st.pyplot(plot_classic(df, 'eur_12m'))
# st.line_chart(df.loc[:,'eur_12m'])

# Information about this 
year = 2021
st.markdown(f"## Evolution in {year}")
st.line_chart(df.loc[f'{year}', 'eur_12m'])


start = datetime(2020, 1, 1)
end = datetime(2020, 7, 1)
st.pyplot(compare_years(df, start, end ))

# compa

