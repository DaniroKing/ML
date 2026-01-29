import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import anderson, chi2_contingency, fisher_exact
from sklearn.preprocessing import StandardScaler

# ЧАСТЬ 1:
# Загрузка и объединение данных 
df_cad = pd.read_excel('курс доллар канада 1.xlsx', sheet_name='RC').rename(columns={'curs': 'CAD', 'data': 'date'})
df_eur = pd.read_excel('курс евро 1.xlsx', sheet_name='RC').rename(columns={'curs': 'EUR', 'data': 'date'})
df_usd = pd.read_excel('курс доллар сша 1.xlsx', sheet_name='RC').rename(columns={'curs': 'USD', 'data': 'date'})
df_currency = df_cad[['date', 'CAD']].merge(df_eur[['date', 'EUR']], on='date').merge(df_usd[['date', 'USD']], on='date')

# Стандартизация данных 
scaler = StandardScaler()
currency_standardized = scaler.fit_transform(df_currency[['USD', 'EUR', 'CAD']])
df_currency[['USD', 'EUR', 'CAD']] = currency_standardized

# Анализ нормальности с помощью теста Андерсона-Дарлинга
print("\nАНАЛИЗ НОРМАЛЬНОСТИ ДАННЫХ:")

for currency in ['USD', 'EUR', 'CAD']:
    data = df_currency[currency]
    result = anderson(data)
    print(result)
    print(f"\n{currency}:")
    print(f"  Статистика теста: {result.statistic:.4f}")
    
    if result.statistic > result.critical_values[2]:
        print(f"  ВЫВОД: распределение ОТЛИЧАЕТСЯ от нормального")
    else:
        print(f"  ВЫВОД: распределение НЕ ОТЛИЧАЕТСЯ от нормального")

# Графики
for currency in ['USD', 'EUR', 'CAD']:
    data = df_currency[currency]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Анализ  данных: {currency}')
    
    # Гистограмма
    ax1.hist(data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax1.axvline(np.mean(data), color='red', linestyle='--', label=f'Среднее')
    ax1.axvline(np.median(data), color='green', linestyle='--', label=f'Медиана')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Корреляционный анализ 
print("\nКОРРЕЛЯЦИИ МЕЖДУ ВАЛЮТАМИ:")
corr_spearman = df_currency[['USD', 'EUR', 'CAD']].corr(method='spearman')
print(corr_spearman)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', center=0, fmt='.3f')
plt.title('Корреляции Спирмена между валютами')
plt.show()

# Сравнение методов корреляции
print("\nСРАВНЕНИЕ МЕТОДОВ КОРРЕЛЯЦИИ:")
currency_pairs = [('USD', 'EUR'), ('USD', 'CAD'), ('EUR', 'CAD')]

for curr1, curr2 in currency_pairs:
    data1 = df_currency[curr1]
    data2 = df_currency[curr2]
    
    pearson = stats.pearsonr(data1, data2)[0]
    spearman = stats.spearmanr(data1, data2)[0]
    kendall = stats.kendalltau(data1, data2)[0]
    
    print(f"\n{curr1}-{curr2}:")
    print(f"  Пирсон = {pearson:.3f}")
    print(f"  Спирмен = {spearman:.3f}")
    print(f"  Кендалл = {kendall:.3f}")
    print(f"  Рекомендуется: Спирмен")

# ЧАСТЬ 2:
# Загрузка и фильтрация данных опроса
df_survey = pd.read_csv('2016-FCC-New-Coders-Survey-Data.csv')
columns_needed = ['EmploymentField', 'EmploymentStatus', 'Gender', 'JobPref', 'JobWherePref', 'MaritalStatus', 'Income']
df_analysis = df_survey[columns_needed]

# Фильтрация: только male/female и без пропусков
df_analysis = df_analysis[df_analysis['Gender'].isin(['male', 'female'])].dropna()
print(f"Данные после фильтрации: {len(df_analysis)} записей")

# Стандартизация доходов
if 'Income' in df_analysis.columns:
    income_clean = df_analysis['Income'].replace([0, -1], np.nan).dropna()
    income_standardized = StandardScaler().fit_transform(income_clean.values.reshape(-1, 1)).flatten()
    
    print(f"\nСТАТИСТИКА ДОХОДОВ ПОСЛЕ СТАНДАРТИЗАЦИИ:")
    print(f"Среднее: {np.mean(income_standardized):.2f}")
    print(f"Стандартное отклонение: {np.std(income_standardized):.2f}")
    print(f"Минимум: {np.min(income_standardized):.2f}")
    print(f"Максимум: {np.max(income_standardized):.2f}")

# Анализ связей между переменными
print("\nАНАЛИЗ СВЯЗЕЙ МЕЖДУ ПЕРЕМЕННЫМИ:")

pairs = [
    ('Gender', 'JobPref'),
    ('Gender', 'JobWherePref'), 
    ('JobWherePref', 'MaritalStatus'),
    ('EmploymentField', 'JobWherePref'),
    ('EmploymentStatus', 'JobWherePref')
]

for col1, col2 in pairs:
    print(f"\n{col1} and {col2}:")
    
    contingency = pd.crosstab(df_analysis[col1], df_analysis[col2])
    contingency = contingency.loc[contingency.sum(axis=1) >= 5]
    contingency = contingency.loc[:, contingency.sum(axis=0) >= 5]
    
    print("  Таблица сопряженности:")
    print(contingency)
    
    if contingency.size < 20:
        odds_ratio, p_value = fisher_exact(contingency)
        print(f"  Точный критерий Фишера: p-value = {p_value:.4f}")
    else:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        print(f"  Хи-квадрат Пирсона: p-value = {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  Статистически значима")
    else:
        print(f"  Статистически не значима")
