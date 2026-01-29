import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# Задание 1 и 2
df = pd.read_csv("C:/Users/Даниил/OneDrive/Desktop/МИФИ/ML/Lab_3/wine_quality_merged.csv")
print(df)

# Pandas хорошо определяет числа, но 'type' лучше сделать категорией
df['type'] = df['type'].astype('category')

# Вывод информации о типах данных для проверки
print("Информация по DataFrame и типах данных")
print(df.info())

# Задание 3
# Выбираем только числовые признаки для стандартизации
# Исключаем 'quality' (целевая переменная) и 'type' (категориальная строка)
features_to_scale = df.columns.drop(['quality', 'type'])

# Инициализируем стандартизатор
scaler = StandardScaler()

#Создаем копию датафрейма, чтобы не потерять оригинал (хорошая практика)
df_scaled = df.copy()

#Применяем стандартизацию только к выбранным столбцам
# fit_transform вычисляет среднее/отклонение и сразу преобразует данные
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print("--- Стандартизация выполнена ---")
print("Первые 5 строк стандартизированного DataFrame:")
print(df_scaled.head())

#Задание 4
# Формирование матрицы признаков (X) и вектора ответов (y)
# X: Берем наши стандартизированные данные.
# Важно: Мы пока не закодировали столбец 'type' (текст), поэтому для обучения
# берем только числовые признаки, которые мы стандартизировали в прошлом шаге.
X = df_scaled.drop(['quality', 'type'], axis=1)
# y: Это наша целевая переменная (оценка качества)
y = df['quality']
# Разделение на обучение и тест
# test_size=0.4 задает размер тестовой выборки (40%), значит на обучение останется 60%.
# random_state=42 — это "зерно" случайности. Нужно, чтобы при каждом запуске
# данные разбивались одинаково (для воспроизводимости результата).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# Вывод размеров получившихся таблиц для проверки
print("\n Разделение данных завершено")
print(f"Кол-во строк в наборе: {len(df)}")
print(f"Обучающая выборка (X_train): {X_train.shape} (60%)")
print(f"Тестовая выборка  (X_test):  {X_test.shape} (40%)")

# Задание 5
# Список конфигураций для проверки
# Мы берём разные ядра и значения C(параметр регуляризации), чтобы найти лучшее сочетание

configs = [
    {'label': '1. Linear Kernel (Базовый)', 'params': {'kernel': 'linear', 'C': 1.0}}, # Линейное ядро (предполагается линейная зависимость)
    {'label': '2. RBF Kernel (Стандарт)',   'params': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}}, # Радиально-базисная функция (для большинства задач, особенно когда неизвестна природа данных)
    {'label': '3. RBF + High C (Жесткий)',  'params': {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'}}, #  RBF, но с жёсткой классификацией (данные чистые и нужно максимально точное разделение)
    {'label': '4. Poly Kernel (Степень 3)', 'params': {'kernel': 'poly', 'degree': 3, 'C': 1.0}}, # Полиномиальное ядро (есть основания предполагать полиномиальную зависимость в данных)
    {'label': '5. Sigmoid (Эксперимент)',   'params': {'kernel': 'sigmoid', 'C': 1.0, 'coef0': 0.0}} # Сигмоидное ядро (когда другие ядра не работают)
]

print("\n SVM обучение ")
best_accuracy = 0
best_config = None
for config in configs:
    print(f"Используем: {config['label']}...")

    # Инициализация модели с текущими параметрами
    # config['params'] распаковывает словарь параметров внутрь функции

    clf = SVC(**config['params'], random_state=42) # random_state=42 фиксирует случайное начальное состояние для воспроизводимости результатов

    #  Обучение
    clf.fit(X_train, y_train)

    # Применяем обученную модель к тестовым данным
    y_pred = clf.predict(X_test)

    # Расчет точности
    acc = accuracy_score(y_test, y_pred)
    print(f" = Точность: {acc:.4f}")

    # Запоминаем лучшую модель
    if acc > best_accuracy:
        best_accuracy = acc
        best_config = config

print(f"ЛУЧШИЙ РЕЗУЛЬТАТ: {best_accuracy:.4f}")
print(f"Параметры: {best_config['params']}")

# Задание 6
# Настройка PCA
# n_components=0.95 означает: "Оставляем столько компонент, чтобы сохранить 95% информации (дисперсии) исходных данных, 
# Это автоматически подберет оптимальное количество.

pca = PCA(n_components=0.95)

# Обучение PCA и трансформация данных
# fit(вычисляет главные компоненты на обучающих данных) делаем только на train, а transform(проецирует обучающие данные на эти компоненты) на обоих

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Узнаем, сколько компонент осталось
n_components = X_train_pca.shape[1]
print(f"Исходное количество признаков: {X_train.shape[1]}")
print(f"Количество компонент после PCA (для сохранения 95% дисперсии): {n_components}")

# Повторное обучение SVM 
# Используем параметры наилучшего: Kernel=rbf, C=10

clf_pca = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
print("Обучение SVM на данных после PCA")
clf_pca.fit(X_train_pca, y_train)

# 4. Предположение и оценка

y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print(f"= Точность (PCA): {acc_pca:.4f}")

# Сравнение
delta = acc_pca - 0.5633  # Сравниваем с предыдущим результатом
print(f"Изменение точности: {delta:.4f}")