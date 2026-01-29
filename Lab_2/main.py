import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import anderson, chi2_contingency, fisher_exact

# ЧАСТЬ 1: АНАЛИЗ КУРСОВ ВАЛЮТ
print("\nЧАСТЬ 1: АНАЛИЗ КУРСОВ ВАЛЮТ")

# Загрузка и объединение данных валют
df_cad = pd.read_excel('курс доллар канада 1.xlsx', sheet_name='RC').rename(columns={'curs': 'CAD', 'data': 'date'})
df_eur = pd.read_excel('курс евро 1.xlsx', sheet_name='RC').rename(columns={'curs': 'EUR', 'data': 'date'})
df_usd = pd.read_excel('курс доллар сша 1.xlsx', sheet_name='RC').rename(columns={'curs': 'USD', 'data': 'date'})

df_currency = df_cad[['date', 'CAD']].merge(df_eur[['date', 'EUR']], on='date').merge(df_usd[['date', 'USD']], on='date')

print(f"Данные валют: {len(df_currency)} записей")

# Анализ нормальности с помощью теста Андерсона-Дарлинга
print("\nАНАЛИЗ НОРМАЛЬНОСТИ (ТЕСТ АНДЕРСОНА-ДАРЛИНГА):")

# Цикл анализа нормальности для каждой валюты
for currency in ['USD', 'EUR', 'CAD']:
    data = df_currency[currency]
    
    # Тест Андерсона-Дарлинга
    result = anderson(data)
    statistic = result.statistic # расчетное значение тестовой статистики
    critical_values = result.critical_values # пороговые значения для сравнения
    significance_levels = result.significance_level # уровни значимости 
    
    print(f"\n{currency}:")
    print(f"  Статистика теста: {statistic:.4f}")
    
    # Проверка на разных уровнях значимости
    for i, sl in enumerate(significance_levels):
        cv = critical_values[i]
        if statistic > cv:
            print(f"  На уровне значимости {sl}%: распределение НЕ нормальное (статистика {statistic:.4f} > критического {cv:.4f})")
        else:
            print(f"  На уровне значимости {sl}%: распределение нормальное (статистика {statistic:.4f} <= критического {cv:.4f})")
    
    # Основной вывод (на уровне 5%)
    if statistic > critical_values[2]: 
        print(f"ОСНОВНОЙ ВЫВОД: распределение ОТЛИЧАЕТСЯ от нормального")
    else:
        print(f"ОСНОВНОЙ ВЫВОД: распределение НЕ ОТЛИЧАЕТСЯ от нормального")

# Визуальный анализ для подтверждения
print("\nВИЗУАЛЬНЫЙ АНАЛИЗ ДЛЯ ПОДТВЕРЖДЕНИЯ:")
for currency in ['USD', 'EUR', 'CAD']:
    data = df_currency[currency]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Визуальный анализ: {currency}')
    
    # Гистограмма
    ax1.hist(data, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.axvline(np.mean(data), color='red', linestyle='--', label=f'Среднее: {np.mean(data):.2f}')
    ax1.axvline(np.median(data), color='green', linestyle='--', label=f'Медиана: {np.median(data):.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Корреляционный анализ 
print("\nКОРРЕЛЯЦИИ МЕЖДУ ВАЛЮТАМИ (МЕТОД СПИРМЕНА):")
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
    print(f"  Рекомендуется: Спирмен (данные не нормальные)")

# ЧАСТЬ 2: АНАЛИЗ ОПРОСА ПРОГРАММИСТОВ

print("\nЧАСТЬ 2: АНАЛИЗ ОПРОСА ПРОГРАММИСТОВ")

# Загрузка и фильтрация данных опроса
df_survey = pd.read_csv('2016-FCC-New-Coders-Survey-Data.csv')
columns_needed = ['EmploymentField', 'EmploymentStatus', 'Gender', 'JobPref', 'JobWherePref', 'MaritalStatus', 'Income']
df_analysis = df_survey[columns_needed]

# Фильтрация: только male/female и без пропусков
df_analysis = df_analysis[df_analysis['Gender'].isin(['male', 'female'])].dropna()
print(f"Данные после фильтрации: {len(df_analysis)} записей")

# Анализ связей между переменными
print("\nАНАЛИЗ СВЯЗЕЙ МЕЖДУ ПЕРЕМЕННЫМИ:")

# Список пар для анализа
pairs = [
    ('Gender', 'JobPref'),
    ('Gender', 'JobWherePref'), 
    ('JobWherePref', 'MaritalStatus'),
    ('EmploymentField', 'JobWherePref'),
    ('EmploymentStatus', 'JobWherePref')
]

# Цикл анализа каждой пары переменных
for col1, col2 in pairs:
    print(f"\n{col1} and {col2}:")
    
    # Создание таблицы сопряженности
    contingency = pd.crosstab(df_analysis[col1], df_analysis[col2])
    
    # Фильтруем редкие категории (менее 5 наблюдений)
    contingency = contingency.loc[contingency.sum(axis=1) >= 5]
    contingency = contingency.loc[:, contingency.sum(axis=0) >= 5]
    
    # Проверяем, достаточно ли данных после фильтрации
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        print("  Недостаточно данных для анализа")
        continue
    
    print("  Таблица сопряженности:")
    print(contingency)
    
    # Выбор критерия в зависимости от размера таблицы
    if contingency.size < 20:
        # Используем точный критерий Фишера для маленьких таблиц
        odds_ratio, p_value = fisher_exact(contingency)
        print(f"  Точный критерий Фишера: p-value = {p_value:.4f}")
    else:
        # Используем хи-квадрат Пирсона для больших таблиц
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        print(f"  Хи-квадрат Пирсона: p-value = {p_value:.4f}")
    
    # Интерпретация результата
    if p_value < 0.05:
        print(f"  Статистически значима")
    else:
        print(f"  Статистически не значима")

print("\nАНАЛИЗ ЗАВЕРШЕН")