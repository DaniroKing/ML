
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('kc_house_data.csv', parse_dates=['date'])

df_selected = df[['date', 'price', 'yr_built', 'yr_renovated', 'sqft_living', 'condition']].copy()
df_selected['real_year'] = df_selected[['yr_built', 'yr_renovated']].max(axis=1)

df_sorted = df_selected.sort_values(['real_year', 'sqft_living', 'condition']).reset_index(drop=True)

year_counts = df_sorted['real_year'].value_counts().sort_values(ascending=True)

top_2 = year_counts.tail(2).index
bottom_2 = year_counts.head(2).index
selected_years = top_2.union(bottom_2)

df_filtered = df_sorted[~df_sorted['real_year'].isin(selected_years)]

condition_counts = df_filtered['condition'].value_counts(ascending=False)

scaler = MinMaxScaler()
df_normalized = df_sorted.copy()
df_normalized[['real_year_norm', 'condition_norm']] = scaler.fit_transform(df_sorted[['real_year', 'condition']])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

stats.probplot(df_sorted['real_year'], dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot Real Year')

stats.probplot(df_sorted['condition'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot Condition')

