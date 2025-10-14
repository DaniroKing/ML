
"""
Лабораторная работа: Анализ данных о продажах домов (Вариант 4)
Раджабов Д.А.
Богданович Г.Б.
C22-701
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

print("ЛАБОРАТОРНАЯ РАБОТА: АНАЛИЗ ДАННЫХ О ПРОДАЖАХ ДОМОВ")
print("=" * 50)

# ЗАГРУЗКА ДАННЫХ
print("\n1. ЗАГРУЗКА ДАННЫХ")
df = pd.read_csv('kc_house_data.csv', parse_dates=['date'])
print(f"Загружено {len(df)} записей")

# ВЫБОР ПЕРЕМЕННЫХ
print("\n2. ВЫБОР ПЕРЕМЕННЫХ")
df_selected = df[['date', 'price', 'yr_built', 'yr_renovated', 'sqft_living', 'condition']].copy()
df_selected['real_year'] = df_selected[['yr_built', 'yr_renovated']].max(axis=1)
print(f" Создана real_year, записей: {len(df_selected)}")

# СОРТИРОВКА
print("\n3. СОРТИРОВКА ДАННЫХ")
df_sorted = df_selected.sort_values(['real_year', 'sqft_living', 'condition']).reset_index(drop=True)
print("Данные отсортированы")

# АНАЛИЗ ПО ГОДАМ
print("\n4. АНАЛИЗ ПО REAL_YEAR")
year_counts = df_sorted['real_year'].value_counts().sort_values(ascending=True)
print(f"Уникальных годов: {len(year_counts)}")

# ВЫБОР 4 ЛЕТ
print("\n5. ВЫБОР 4 ЗНАЧЕНИЙ ДЛЯ REAL_YEAR")
top_2 = year_counts.tail(2).index
bottom_2 = year_counts.head(2).index
selected_years = top_2.union(bottom_2)
print(f"Выбраны года: {list(selected_years)}")

# ФИЛЬТРАЦИЯ
print("\n6. ФИЛЬТРАЦИЯ ДАННЫХ")
df_filtered = df_sorted[~df_sorted['real_year'].isin(selected_years)]
print(f"Удалено записей: {len(df_sorted) - len(df_filtered)}")

# АНАЛИЗ ПО CONDITION
print("\n7. АНАЛИЗ ПО CONDITION")
condition_counts = df_filtered['condition'].value_counts(ascending=False)
print("Распределение condition:", dict(condition_counts))

# НОРМАЛИЗАЦИЯ
print("\n8. НОРМАЛИЗАЦИЯ ДАННЫХ")
scaler = MinMaxScaler()
df_normalized = df_sorted.copy()
df_normalized[['real_year_norm', 'condition_norm']] = scaler.fit_transform(df_sorted[['real_year', 'condition']])
print("Нормализация завершена")

# ГРАФИКИ
print("\n9. ПОСТРОЕНИЕ ГРАФИКОВ")

# Распределения
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Real Year
sns.histplot(df_sorted['real_year'], kde=True, ax=axes[0, 0], stat='density')
axes[0, 0].set_title('Распределение Real Year')

stats.probplot(df_sorted['real_year'], dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot Real Year')

# Condition
sns.histplot(df_sorted['condition'], kde=True, ax=axes[1, 0], stat='density')
axes[1, 0].set_title('Распределение Condition')

stats.probplot(df_sorted['condition'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot Condition')

plt.tight_layout()
plt.show()

# Регрессия
print("\n10. ГРАФИК ЛИНЕЙНОЙ РЕГРЕССИИ")
plt.figure(figsize=(10, 6))
sns.regplot(x='real_year', y='condition', data=df_sorted, scatter_kws={'alpha':0.3}, line_kws={'color': 'red'})
plt.title('Real Year vs Condition')
plt.show()

correlation = df_sorted['real_year'].corr(df_sorted['condition'])
print(f"Коэффициент корреляции: {correlation:.3f}")

print("\n ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ!")