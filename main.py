import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt


def categorize_age(age):
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teenager'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Elderly'

def extract_ticket_prefix(ticket):
    parts = ticket.split(' ')
    if len(parts) > 1:
        return parts[0].strip()
    else:
        return 'Unknown'


def extract_deck(cabin):
    if pd.isna(cabin):
        return 'Unknown'
    else:
        return cabin[0]


def extract_title(name):
    return name.split(",")[1].split(".")[0].strip()


def preprocess_data(input_file):
    DAT = pd.read_csv(input_file)

    DAT["Embarked"].fillna(DAT["Embarked"].mode()[0])

    DAT['Title'] = DAT['Name'].apply(extract_title)

    DAT['Title'] = DAT['Title'].replace(['Mlle', 'Ms'], 'Miss')
    DAT['Title'] = DAT['Title'].replace('Mme', 'Mrs')

    DAT['Deck'] = DAT['Cabin'].apply(extract_deck)
    DAT = DAT.drop(columns=["Cabin"])

    categorical_cols = ['Sex', 'Embarked', 'Deck', 'Pclass', 'Title']
    DAT = pd.get_dummies(DAT, columns=categorical_cols, drop_first=True)


    features_SibSp_Parch = DAT[["SibSp", "Parch"]]
    tariff = DAT[['Fare']]

    scaler_minmax = MinMaxScaler()
    normalized_features = scaler_minmax.fit_transform(features_SibSp_Parch)

    scaler_robust = RobustScaler()
    standardized_tariff = scaler_robust.fit_transform(tariff)

    processed_df = pd.DataFrame(
        np.hstack([normalized_features, standardized_tariff]),
        columns=['SibSp_normalized', 'Parch_normalized', 'tariff_standardized']
    )

    columns_to_drop = ["SibSp", "Parch", "Fare", "Ticket", 'Name']
    DAT.drop(columns=columns_to_drop, inplace=True)

    combined_df = pd.concat([DAT.reset_index(drop=True), processed_df.reset_index(drop=True)], axis=1)

    DAT_filled = combined_df.copy()

    known_age = DAT_filled[DAT_filled['Age'].notnull()]
    unknown_age = DAT_filled[DAT_filled['Age'].isnull()].drop(columns=['Age'])

    X_train_age = known_age.drop(columns=['Age'])
    y_train_age = known_age['Age']

    rf_age = RandomForestRegressor(random_state=42)
    rf_age.fit(X_train_age, y_train_age)

    predicted_age = rf_age.predict(unknown_age)

    DAT_filled.loc[DAT_filled['Age'].isnull(), 'Age'] = predicted_age

    # Создаем новый признак 'AgeCategory' на основе столбца 'Age'
    DAT_filled['AgeCategory'] = DAT_filled['Age'].apply(categorize_age)
    DAT_filled = pd.get_dummies(DAT_filled, columns=['AgeCategory'], drop_first=True)
    # age = DAT_filled[['Age']]
    # scaler_standard = StandardScaler()
    # standardized_age = scaler_standard.fit_transform(age)
    #
    # processed_df2 = pd.DataFrame(
    #     np.hstack([standardized_age]),
    #     columns=['Norm_Age']
    # )

    columns_to_drop = ["Age"]
    DAT_filled.drop(columns=columns_to_drop, inplace=True)

    # combined_df = pd.concat([DAT_filled.reset_index(drop=True), processed_df2.reset_index(drop=True)], axis=1)

    return DAT_filled

trd = pd.read_csv('train.csv')
tsd = pd.read_csv('test.csv')
td = pd.concat([trd, tsd], ignore_index=True, sort = False)
td.to_csv('output.csv', index=False)

input_file = "train.csv"
combined_df = preprocess_data(input_file)
X = combined_df.drop(columns=['Survived'])  # Удаляем целевую переменную, если она есть
y = combined_df['Survived']  # Целевая переменная
# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
# Определение сетки параметров
best_params = {
    'colsample_bytree': 0.5,
    'gamma': 0,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 3
}
# Создаем объект модели XGBoost
model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Точность модели на тестовой выборке: {accuracy:.4f}")

# feature_names = X.columns.values
# # Получаем важность признаков
# importances = model.feature_importances_
#
# # Сортируем признаки по их важности
# indices = np.argsort(importances)[::-1]
#
# # Выводим важность признаков по их названиям
# print("Важность признаков:")
# for i in range(X.shape[1]):
#     print(f"{i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]}")
#
# # Визуализируем важность признаков
# plt.figure(figsize=(10, 6))
# plt.title("Важность признаков")
# plt.bar(range(X.shape[1]), importances[indices], align="center")
# plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
# plt.xlim([-1, X.shape[1]])
# plt.tight_layout()
# plt.show()

features_list = combined_df.columns.tolist()
print("Список признаков:")
print(features_list)

input_file_kag = "output.csv"
df_kag = preprocess_data(input_file_kag)
known_Survived = df_kag[df_kag['Survived'].notnull()]
unknown_Survived = df_kag[df_kag['Survived'].isnull()].drop(columns=['Survived'])

X_train_Survived = known_Survived.drop(columns=['Survived'])
y_train_Survived = known_Survived['Survived']

rf_Survived = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False)
rf_Survived.fit(X_train_Survived, y_train_Survived)

predicted_Survived = rf_Survived.predict(unknown_Survived)

df_kag.loc[df_kag['Survived'].isnull(), 'Survived'] = predicted_Survived
df_kag = df_kag[['PassengerId', 'Survived']]
df_kag.Survived = df_kag.Survived.astype(int)
df_kag.iloc[891:].to_csv('kag.csv', index=False)

