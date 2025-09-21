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
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

RANDOM_STATE = 42

train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

train_data['Target'] = train_data['Дата отмены'].notnull().astype(int)
train_data.drop(['Дата отмены', 'Статус брони'], axis=1, inplace=True)

train_data['is_train'] = 1
test_data['is_train'] = 0
test_data['Target'] = np.nan
data = pd.concat([train_data, test_data], sort=False, ignore_index=True)

data['Внесена предоплата'] = data['Внесена предоплата'].fillna(0)
data['Категория номера'] = data['Категория номера'].fillna('Не указана')
data['Гостей'] = data['Гостей'].fillna(data['Гостей'].median())
data['Номеров'] = data['Номеров'].fillna(1)
data['Ночей'] = data['Ночей'].fillna(1)

date_columns = ['Дата бронирования', 'Заезд', 'Выезд']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], format='%d.%m.%Y %H:%M', errors='coerce')

data['Время до заезда'] = (data['Заезд'] - data['Дата бронирования']).dt.days.fillna(-1)
data['Длительность пребывания'] = (data['Выезд'] - data['Заезд']).dt.days.fillna(-1)
data['Месяц бронирования'] = data['Дата бронирования'].dt.month
data['Месяц заезда'] = data['Заезд'].dt.month
data['День недели бронирования'] = data['Дата бронирования'].dt.weekday
data['День недели заезда'] = data['Заезд'].dt.weekday

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

ru_holidays = holidays.Russia()
data['Праздник при бронировании'] = data['Дата бронирования'].dt.date.isin(ru_holidays).astype(int)
data['Праздник при заезде'] = data['Заезд'].dt.date.isin(ru_holidays).astype(int)

yuan_rate = pd.read_excel('yuan_rate.xlsx', decimal=',')
yuan_rate['data'] = pd.to_datetime(yuan_rate['data'], format='%d.%m.%Y')
yuan_rate = yuan_rate[['data', 'curs']].rename(columns={'data': 'date', 'curs': 'yuan_rate'})

usd_rate = pd.read_excel('usd_rate.xlsx', decimal=',')
usd_rate['data'] = pd.to_datetime(usd_rate['data'], format='%d.%m.%Y')
usd_rate = usd_rate[['data', 'curs']].rename(columns={'data': 'date', 'curs': 'usd_rate'})

currency_rates = pd.merge(yuan_rate, usd_rate, on='date', how='outer')
currency_rates.sort_values('date', inplace=True)
currency_rates.ffill(inplace=True)

data['Дата бронирования_дата'] = data['Дата бронирования'].dt.normalize()
data = data.merge(currency_rates, how='left', left_on='Дата бронирования_дата', right_on='date')
data.drop(['date', 'Дата бронирования_дата'], axis=1, inplace=True)

data['yuan_rate'] = data['yuan_rate'].ffill().bfill()
data['usd_rate'] = data['usd_rate'].ffill().bfill()

with open('tourism_demand.json', 'r', encoding='utf-8') as f:
    tourism_data = json.load(f)

tourism_df = []
for year, quarters in tourism_data.items():
    for quarter, metrics in quarters.items():
        tourism_df.append({
            'year': int(year),
            'quarter': quarter,
            'tourism_balance': metrics['balance']
        })
tourism_df = pd.DataFrame(tourism_df)

def get_quarter(month):
    if month in [1, 2, 3]:
        return 'I_quarter'
    elif month in [4, 5, 6]:
        return 'II_quarter'
    elif month in [7, 8, 9]:
        return 'III_quarter'
    elif month in [10, 11, 12]:
        return 'IV_quarter'
    else:
        return 'Unknown'

data['year'] = data['Дата бронирования'].dt.year
data['quarter'] = data['Дата бронирования'].dt.month.apply(get_quarter)

data = data.merge(tourism_df, how='left', on=['year', 'quarter'])
data['tourism_balance'] = data['tourism_balance'].ffill().bfill()
data.drop(['year', 'quarter'], axis=1, inplace=True)

with open('weather.json', 'r', encoding='utf-8') as f:
    weather_data = json.load(f)

weather_df = pd.DataFrame(weather_data)
weather_df.rename(columns={'tempreture': 'temperature'}, inplace=True)

month_mapping = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}
weather_df['month'] = weather_df['month'].str.lower().map(month_mapping)
weather_df['day'] = weather_df['day'].astype(int)
weather_df['year'] = 2023
weather_df['date'] = pd.to_datetime(weather_df[['year', 'month', 'day']])

def parse_temperature(temp_str):
    temp_str = temp_str.replace('°', '').replace('+', '').replace('−', '-')
    return int(temp_str)

weather_df['temperature'] = weather_df['temperature'].apply(parse_temperature)
weather_df = weather_df[['date', 'temperature', 'weather']]

data['Заезд_дата'] = data['Заезд'].dt.normalize()
data = data.merge(weather_df, how='left', left_on='Заезд_дата', right_on='date')
data.loc[~data['Гостиница'].isin([1, 2]), ['temperature', 'weather']] = np.nan
data['temperature'] = data['temperature'].fillna(data['temperature'].mean())
data['weather'] = data['weather'].fillna('NoData')
data.drop(['Дата бронирования', 'Заезд', 'Выезд', 'Заезд_дата', 'date'], axis=1, inplace=True)

categorical_features = ['Способ оплаты', 'Источник', 'Категория номера', 'Гостиница',
                        'Сезон бронирования', 'Сезон заезда', 'weather']
numerical_features = ['Внесена предоплата', 'Гостей', 'Номеров', 'Ночей',
                      'Время до заезда', 'Длительность пребывания',
                      'Месяц бронирования', 'Месяц заезда',
                      'День недели бронирования', 'День недели заезда',
                      'temperature', 'yuan_rate', 'usd_rate', 'tourism_balance']

train_df = data[data['is_train'] == 1].drop(['is_train'], axis=1)
test_df = data[data['is_train'] == 0].drop(['is_train', 'Target'], axis=1)

y = train_df['Target']
X = train_df.drop(['Target', '№ брони'], axis=1)
X_test = test_df.drop(['№ брони'], axis=1)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

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

class_counts = Counter(y)
imbalance_ratio = class_counts[0] / class_counts[1]
model.set_params(scale_pos_weight=imbalance_ratio)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

clf.fit(X, y)

test_pred = clf.predict(X_test)

np.savetxt('submission.csv', test_pred, fmt='%d', encoding='utf-8')
print("Файл submission.csv сохранён.")
