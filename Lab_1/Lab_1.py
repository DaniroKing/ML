import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# 1. ЗАГРУЗКА ДАННЫХ
print("\n ЗАГРУЗКА ДАННЫХ")
df = pd.read_csv('kc_house_data.csv', parse_dates=['date'])
print(f"Загружено {len(df)} записей")

# 2. ВЫБОР ПЕРЕМЕННЫХ
print("\n ВЫБОР ПЕРЕМЕННЫХ")
df_selected = df[['date', 'price', 'yr_built', 'yr_renovated', 'sqft_living', 'condition']].copy()
df_selected['real_year'] = df_selected[['yr_built', 'yr_renovated']].max(axis=1)
print(f"Создана real_year, записей: {len(df_selected)}")

# 3. СОРТИРОВКА
print("\n СОРТИРОВКА ДАННЫХ")
df_sorted = df_selected.sort_values(['real_year', 'sqft_living', 'condition']).reset_index(drop=True)
print("Данные отсортированы")

# 4. АНАЛИЗ ПО ГОДАМ
print("\n АНАЛИЗ ПО REAL_YEAR")
year_counts = df_sorted['real_year'].value_counts().sort_values(ascending=True)
print(f"Уникальных годов: {len(year_counts)}")

# 5. ВЫБОР 4 ЛЕТ
print("\n ВЫБОР 4 ЗНАЧЕНИЙ ДЛЯ REAL_YEAR")
top_2 = year_counts.tail(2).index
bottom_2 = year_counts.head(2).index
selected_years = top_2.union(bottom_2)
print(f"Выбраны года: {list(selected_years)}")

# 6. ФИЛЬТРАЦИЯ
print("\n ФИЛЬТРАЦИЯ ДАННЫХ")
df_filtered = df_sorted[~df_sorted['real_year'].isin(selected_years)]
print(f"Удалено записей: {len(df_sorted) - len(df_filtered)}")

# 7. АНАЛИЗ ПО CONDITION
print("\n АНАЛИЗ ПО CONDITION")
condition_counts = df_filtered['condition'].value_counts(ascending=False)
print("Распределение condition:", dict(condition_counts))

# 8. НОРМАЛИЗАЦИЯ ДАННЫХ
print("\n НОРМАЛИЗАЦИЯ ДАННЫХ")
scaler = MinMaxScaler()
df_normalized = df_sorted.copy()
df_normalized[['real_year_norm', 'condition_norm']] = scaler.fit_transform(
    df_sorted[['real_year', 'condition']]
)
print("Нормализация завершена")

# 10. ДИАГРАММА РАССЕИВАНИЯ 
print("\n ДИАГРАММА РАССЕИВАНИЯ REAL_YEAR VS CONDITION")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='real_year', y='condition', data=df_sorted, alpha=0.3)
plt.title('Real Year vs Condition')
plt.xlabel('Real Year')
plt.ylabel('Condition')
plt.grid(True)
plt.show()

