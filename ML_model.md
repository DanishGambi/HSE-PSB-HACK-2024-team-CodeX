# Подробное описание работы кода

Данный код предназначен для предсказания отмены бронирования в гостинице на основе исторических данных. Он выполняет следующие основные шаги:

1. **Импорт библиотек и модулей.**
2. **Загрузка и объединение данных.**
3. **Предварительная обработка данных.**
4. **Создание новых признаков.**
5. **Обработка данных о погоде.**
6. **Подготовка данных для модели.**
7. **Создание и обучение модели.**
8. **Оценка модели с помощью кросс-валидации.**
9. **Предсказание и сохранение результатов.**

Далее рассмотрим каждый шаг подробно.

## 1. Импорт библиотек и модулей

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import holidays
import json
from collections import Counter

RANDOM_STATE = 42
```

- **pandas** и **numpy**: для работы с данными и числовыми вычислениями.
- **scikit-learn** модули: для предобработки данных, создания модели и оценки её качества.
- **xgboost**: библиотека для градиентного бустинга, используемая для построения модели.
- **holidays**: для определения праздничных дней.
- **json**: для работы с JSON-файлами (данные о погоде).
- **Counter** из **collections**: для подсчёта количества элементов (используется при обработке дисбаланса классов).
- **RANDOM_STATE**: фиксированный сид для воспроизводимости результатов.

## 2. Загрузка и объединение данных

```python
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')
```

- Загружаются тренировочные и тестовые данные из файлов Excel.

```python
train_data['Target'] = train_data['Дата отмены'].notnull().astype(int)
train_data.drop(['Дата отмены', 'Статус брони'], axis=1, inplace=True)
```

- Создаётся целевая переменная `Target`, которая равна 1, если бронирование было отменено, и 0 в противном случае.
- Удаляются столбцы `Дата отмены` и `Статус брони`, так как они больше не нужны.

```python
train_data['is_train'] = 1
test_data['is_train'] = 0
test_data['Target'] = np.nan
data = pd.concat([train_data, test_data], sort=False, ignore_index=True)
```

- Добавляется столбец `is_train` для разделения данных после объединения.
- Целевая переменная в тестовых данных заполняется `NaN`.
- Объединяются тренировочные и тестовые данные для совместной обработки.

## 3. Предварительная обработка данных

```python
data['Внесена предоплата'] = data['Внесена предоплата'].fillna(0)
data['Категория номера'] = data['Категория номера'].fillna('Не указана')
data['Гостей'] = data['Гостей'].fillna(data['Гостей'].median())
data['Номеров'] = data['Номеров'].fillna(1)
data['Ночей'] = data['Ночей'].fillna(1)
```

- Заполняются пропущенные значения в данных:
  - **Внесена предоплата**: пропуски заменяются на 0.
  - **Категория номера**: пропуски заменяются на 'Не указана'.
  - **Гостей**: пропуски заменяются на медианное значение.
  - **Номеров** и **Ночей**: пропуски заменяются на 1.

## 4. Создание новых признаков

```python
date_columns = ['Дата бронирования', 'Заезд', 'Выезд']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], format='%d.%m.%Y %H:%M', errors='coerce')
```

- Преобразуются столбцы с датами в формат `datetime`.

```python
data['Время до заезда'] = (data['Заезд'] - data['Дата бронирования']).dt.days.fillna(-1)
data['Длительность пребывания'] = (data['Выезд'] - data['Заезд']).dt.days.fillna(-1)
```

- Вычисляются новые признаки:
  - **Время до заезда**: количество дней между бронированием и заездом.
  - **Длительность пребывания**: количество дней между заездом и выездом.
- Пропуски заменяются на -1.

```python
data['Месяц бронирования'] = data['Дата бронирования'].dt.month
data['Месяц заезда'] = data['Заезд'].dt.month

data['День недели бронирования'] = data['Дата бронирования'].dt.weekday
data['День недели заезда'] = data['Заезд'].dt.weekday
```

- Извлекаются месяцы и дни недели из дат бронирования и заезда.

```python
def get_season(month):
    if month in [12, 1, 2]:
        return 'Зима'
    elif month in [3, 4, 5]:
        return 'Весна'
    elif month in [6, 7, 8]:
        return 'Лето'
    elif month in [9, 10, 11]:
        return 'Осень'
    else:
        return 'Неизвестно'

data['Сезон бронирования'] = data['Месяц бронирования'].apply(get_season)
data['Сезон заезда'] = data['Месяц заезда'].apply(get_season)
```

- Определяются сезоны для дат бронирования и заезда на основе месяца.

```python
ru_holidays = holidays.Russia()
data['Праздник при бронировании'] = data['Дата бронирования'].dt.date.isin(ru_holidays).astype(int)
data['Праздник при заезде'] = data['Заезд'].dt.date.isin(ru_holidays).astype(int)
```

- Добавляются признаки, указывающие, попадает ли дата бронирования или заезда на праздник.

## 5. Обработка данных о погоде

```python
with open('weather.json', 'r', encoding='utf-8') as f:
    weather_data = json.load(f)

weather_df = pd.DataFrame(weather_data)
weather_df.rename(columns={'tempreture': 'temperature'}, inplace=True)
```

- Загружаются данные о погоде из файла `weather.json`.
- Исправляется опечатка в названии столбца (`tempreture` на `temperature`).

```python
month_mapping = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}
weather_df['month'] = weather_df['month'].str.lower().map(month_mapping)
weather_df['day'] = weather_df['day'].astype(int)
weather_df['year'] = 2023
weather_df['date'] = pd.to_datetime(weather_df[['year', 'month', 'day']])
```

- Преобразуются названия месяцев в числа.
- Создаётся столбец с полной датой.

```python
def parse_temperature(temp_str):
    temp_str = temp_str.replace('°', '').replace('+', '').replace('−', '-')
    return int(temp_str)

weather_df['temperature'] = weather_df['temperature'].apply(parse_temperature)
weather_df = weather_df[['date', 'temperature', 'weather']]
```

- Преобразуются значения температуры в числовой формат.
- Оставляются только необходимые столбцы.

```python
data['Заезд_дата'] = data['Заезд'].dt.normalize()
data = data.merge(weather_df, how='left', left_on='Заезд_дата', right_on='date')
data.loc[~data['Гостиница'].isin([1, 2]), ['temperature', 'weather']] = np.nan
data['temperature'] = data['temperature'].fillna(data['temperature'].mean())
data['weather'] = data['weather'].fillna('NoData')
```

- Объединяются данные о погоде с основными данными по дате заезда.
- Для гостиниц, отличных от 1 и 2, данные о погоде устанавливаются как `NaN`.
- Пропущенные температуры заполняются средним значением, а погодные условия — строкой 'NoData'.

```python
data.drop(['Дата бронирования', 'Заезд', 'Выезд', 'Заезд_дата', 'date'], axis=1, inplace=True)
```

- Удаляются ненужные столбцы с датами.

## 6. Подготовка данных для модели

```python
categorical_features = ['Способ оплаты', 'Источник', 'Категория номера', 'Гостиница',
                        'Сезон бронирования', 'Сезон заезда', 'weather']
numerical_features = ['Внесена предоплата', 'Гостей', 'Номеров', 'Ночей',
                      'Время до заезда', 'Длительность пребывания',
                      'Месяц бронирования', 'Месяц заезда',
                      'День недели бронирования', 'День недели заезда',
                      'temperature']
```

- Определяются списки категориальных и числовых признаков.

```python
train_df = data[data['is_train'] == 1].drop(['is_train'], axis=1)
test_df = data[data['is_train'] == 0].drop(['is_train', 'Target'], axis=1)

y = train_df['Target']
X = train_df.drop(['Target', '№ брони'], axis=1)
X_test = test_df.drop(['№ брони'], axis=1)
```

- Разделяются данные обратно на тренировочные и тестовые наборы.
- Определяются признаки и целевая переменная.

## 7. Создание и обучение модели

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
```

- Создаётся препроцессор для масштабирования числовых признаков и кодирования категориальных признаков методом One-Hot Encoding.

```python
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
    eval_metric='logloss'
)
```

- Создаётся модель XGBoost с заданными гиперпараметрами.

```python
from collections import Counter
class_counts = Counter(y)
imbalance_ratio = class_counts[0] / class_counts[1]
model.set_params(scale_pos_weight=imbalance_ratio)
```

- Рассчитывается соотношение классов для обработки дисбаланса.
- Параметр `scale_pos_weight` устанавливается для компенсации дисбаланса классов.

```python
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])
```

- Создаётся конвейер, объединяющий препроцессор и модель.

## 8. Оценка модели с помощью кросс-валидации

```python
scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
print("Средний ROC-AUC на кросс-валидации:", scores.mean())
```

- Оценивается модель с помощью кросс-валидации на 5 фолдах.
- Вычисляется среднее значение метрики ROC-AUC.

```python
clf.fit(X, y)
```

- Модель обучается на всех тренировочных данных.

## 9. Предсказание и сохранение результатов

```python
test_pred = clf.predict(X_test)
np.savetxt('submission.csv', test_pred, fmt='%d', encoding='utf-8')
print("Файл submission.csv сохранён.")
```

- Предсказания на тестовых данных сохраняются в файл `submission.csv` в требуемом формате.

# Возможности изменения и управления моделью

Для управления моделью и улучшения её качества вы можете внести изменения в следующие части кода:

## 1. **Предобработка данных**

- **Обработка пропусков:**
  - Изменить стратегии заполнения пропусков. Например, вместо заполнения медианой использовать среднее или более сложные методы.
  - Добавить обработку выбросов в числовых данных.

- **Создание новых признаков:**
  - Добавить дополнительные признаки, которые могут быть полезны для модели.
  - Применить методы Feature Engineering, такие как взаимодействие признаков или создание полиномиальных признаков.

- **Удаление ненужных признаков:**
  - Провести анализ корреляции и важности признаков, чтобы удалить малоинформативные или коррелированные признаки.

## 2. **Параметры модели XGBoost**

- **Гиперпараметры модели:**
  - **n_estimators**: количество деревьев в ансамбле. Увеличение может улучшить качество, но увеличит время обучения.
  - **learning_rate**: темп обучения. Меньшее значение требует большего количества деревьев, но может привести к лучшему качеству.
  - **max_depth**: максимальная глубина деревьев. Слишком глубокие деревья могут переобучаться.
  - **subsample**: доля выборки для обучения каждого дерева. Помогает бороться с переобучением.
  - **colsample_bytree**: доля признаков для обучения каждого дерева.
  - **reg_alpha** и **reg_lambda**: параметры регуляризации L1 и L2 соответственно.

- **Пример изменения гиперпараметров:**

  ```python
  model = xgb.XGBClassifier(
      n_estimators=1000,
      learning_rate=0.01,
      max_depth=8,
      subsample=0.8,
      colsample_bytree=0.8,
      reg_alpha=0.1,
      reg_lambda=1,
      random_state=RANDOM_STATE,
      eval_metric='logloss'
  )
  ```

- **Подбор гиперпараметров:**
  - Используйте Grid Search или Randomized Search для автоматического подбора лучших параметров.

  ```python
  from sklearn.model_selection import GridSearchCV

  param_grid = {
      'classifier__n_estimators': [500, 1000],
      'classifier__max_depth': [6, 8],
      'classifier__learning_rate': [0.01, 0.05],
      'classifier__subsample': [0.8, 1.0],
      'classifier__colsample_bytree': [0.8, 1.0]
  }

  grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
  grid_search.fit(X, y)
  print("Лучшие параметры:", grid_search.best_params_)
  clf = grid_search.best_estimator_
  ```

## 3. **Обработка дисбаланса классов**

- **Изменение параметра `scale_pos_weight`:**
  - Текущий параметр рассчитывается как отношение классов. Можно экспериментировать с его значением.

- **Использование методов ресэмплинга:**
  - Применить методы оверсэмплинга (SMOTE) или андерсэмплинга для балансировки данных.

  ```python
  from imblearn.over_sampling import SMOTE

  smote = SMOTE(random_state=RANDOM_STATE)
  X_resampled, y_resampled = smote.fit_resample(X, y)
  clf.fit(X_resampled, y_resampled)
  ```

## 4. **Изменение способа кодирования признаков**

- **One-Hot Encoding vs. Label Encoding:**
  - Можно попробовать заменить One-Hot Encoding на Label Encoding для некоторых признаков, чтобы уменьшить размерность.

- **Использование Target Encoding:**
  - Кодировать категориальные признаки на основе статистики целевой переменной.

## 5. **Изменение алгоритма модели**

- **Попробовать другие модели:**
  - Random Forest, CatBoost, LightGBM.

  ```python
  from lightgbm import LGBMClassifier

  model = LGBMClassifier(
      n_estimators=500,
      learning_rate=0.05,
      max_depth=6,
      random_state=RANDOM_STATE
  )
  ```

- **Ансамблирование моделей:**
  - Объединить предсказания нескольких моделей для улучшения качества.

## 6. **Изменение стратегии кросс-валидации**

- **Изменение числа фолдов:**
  - Увеличение числа фолдов (cv) может дать более стабильную оценку модели.

- **Использование стратифицированной кросс-валидации:**
  - Гарантирует, что распределение классов в фолдах соответствует общему распределению.

  ```python
  from sklearn.model_selection import StratifiedKFold

  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
  scores = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc')
  ```

## 7. **Анализ важности признаков**

- **Вывод важности признаков:**

  ```python
  # Получение имён признаков после One-Hot Encoding
  ohe = clf.named_steps['preprocessor'].named_transformers_['cat']
  feature_names = numerical_features + list(ohe.get_feature_names_out(categorical_features))

  # Получение важности признаков
  importances = clf.named_steps['classifier'].feature_importances_
  feature_importance = pd.DataFrame({'Признак': feature_names, 'Важность': importances})
  feature_importance.sort_values('Важность', ascending=False, inplace=True)
  print("Топ-20 наиболее важных признаков:")
  print(feature_importance.head(20))
  ```

- **Использование важности признаков для отбора:**
  - Удалить менее важные признаки или создать новые на основе важных.

## 8. **Обработка временных признаков**

- **Временные лаги и тренды:**
  - Добавить признаки, отражающие тренды во времени, если данные упорядочены по времени.

- **Сезонность:**
  - Более глубоко исследовать сезонные эффекты и добавить соответствующие признаки.

## 9. **Улучшение обработки данных о погоде**

- **Расширение данных о погоде:**
  - Собрать более подробные данные о погоде для всех гостиниц и за более длительный период.

- **Добавление дополнительных метеорологических показателей:**
  - Влажность, осадки, скорость ветра и т.д.

## 10. **Оптимизация производительности**

- **Снижение размерности:**
  - Использовать методы PCA или SelectKBest для уменьшения количества признаков.

- **Параллелизация вычислений:**
  - Настроить параметр `n_jobs=-1` в моделях для использования всех доступных ядер процессора.

---

# Заключение

Вы можете управлять моделью и её производительностью, внося изменения в различные части кода, начиная от предобработки данных и заканчивая настройкой гиперпараметров модели. Важно проводить эксперименты и оценивать влияние изменений на качество модели, используя метрики и кросс-валидацию.

Рекомендуется вносить изменения постепенно, тщательно анализируя результаты на каждом этапе. Это позволит определить, какие изменения действительно улучшают модель, а какие могут быть избыточными или даже вредными.

Если у вас возникнут дополнительные вопросы или потребуется помощь с реализацией конкретных изменений, пожалуйста, обращайтесь!