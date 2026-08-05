import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Анализ треков Spotify

    Мы анализируем данные музыкального стримингового сервиса, чтобы понять,
    какие треки и жанры наиболее популярны и какие характеристики влияют на
    успех музыки.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.dummy import DummyRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from pathlib import Path

    return (
        DecisionTreeRegressor,
        DummyRegressor,
        KNeighborsRegressor,
        LinearRegression,
        RandomForestRegressor,
        StandardScaler,
        mean_absolute_error,
        np,
        pd,
        plt,
        r2_score,
        sns,
        train_test_split,
        Path
    )


@app.cell
def _():
    base_features = [
        'danceability', 'energy', 'acousticness', 'valence',
        'tempo', 'loudness', 'speechiness',
    ]
    features_with_genre = base_features + ['genre']
    target = 'popularity'
    random_state = 1
    rf_n_estimators = 50
    n_jobs = -1
    ms_per_minute = 60000
    hit_threshold = 70
    mood_happy = 0.7
    mood_sad = 0.3

    return (
        base_features,
        features_with_genre,
        hit_threshold,
        mood_happy,
        mood_sad,
        ms_per_minute,
        n_jobs,
        random_state,
        rf_n_estimators,
        target,
    )


@app.cell
def _(pd, Path):
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "data" / "spotify_data.csv"
    spotify_raw = pd.read_csv(DATA_PATH, index_col=0)
    spotify_raw.head()
    return (spotify_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Очистка данных

    В датасете были пропуски в столбцах `artists`, `album_name`,
    `track_name`.

    - В `artists` 1 пропуск заменили на `"Unknown Artist"`
    - В `album_name` 1 пропуск заменили на `"Unknown Album"`
    - После этого удалили дубликаты по `признакам для будущих моделей`

    В итоге количество строк уменьшилось со 114000 до 83448 — убрали
    повторяющиеся треки с одинаковыми признаками.

    Здесь же сразу переименовываю столбцы (`track_name → song`,
    `artists → artist`, `track_genre → genre`) — в исходном ноутбуке это
    делалось в самом конце, но по факту использовалось (`genre`, `artist`)
    значительно раньше, так что переносим один раз в начало, чтобы всё
    дальше было последовательно и ничего не падало по `KeyError`.

    **Вывод:** после очистки данные стали более надёжными для анализа и
    исключили повторяющиеся записи. Исключая утечку данных, где модель подсматривала бы повторяющийся треки
    """)
    return


@app.cell
def _(base_features, spotify_raw):
    spotify_raw['artists'] = spotify_raw['artists'].fillna("Unknown Artist")
    spotify_raw['album_name'] = spotify_raw['album_name'].fillna("Unknown Album")

    spotify = spotify_raw.drop_duplicates(subset=base_features)
    spotify = spotify.rename(
        columns={
            'track_name': 'song',
            'artists': 'artist',
            'track_genre': 'genre',
        }
    )
    spotify
    return (spotify,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Линейная регрессия.

    Контекст: EDA уже проведён, известно, какие признаки связаны с
    популярностью. Формально выделяем целевую переменную `y` (`popularity`)
    и матрицу признаков `X`, используя колонки: `danceability`, `energy`,
    `acousticness`, `valence`, `tempo`, `loudness`, `speechiness`.
    """)
    return


@app.cell
def _(base_features, spotify, target):
    y_line = spotify[target]
    X_line = spotify[base_features]
    return X_line, y_line


@app.cell
def _(X_line, random_state, train_test_split, y_line):
    train_X_line, test_X_line, train_y_line, test_y_line = train_test_split(X_line, y_line, random_state=random_state)
    return test_X_line, test_y_line, train_X_line, train_y_line


@app.cell
def _(
    LinearRegression,
    base_features,
    mean_absolute_error,
    pd,
    r2_score,
    test_X_line,
    test_y_line,
    train_X_line,
    train_y_line,
):
    model_line = LinearRegression()
    model_line.fit(train_X_line, train_y_line)
    preds_line = model_line.predict(test_X_line)
    print(f'R2-SCORE: {r2_score(test_y_line, preds_line)}')
    print(f'MAE: {mean_absolute_error(test_y_line, preds_line)}')


    print(pd.DataFrame({
        "Features" : base_features,
        "Coeff": model_line.coef_
    }).sort_values('Coeff'))
    return (preds_line,)


@app.cell
def _(
    DummyRegressor,
    mean_absolute_error,
    r2_score,
    test_X_line,
    test_y_line,
    train_X_line,
    train_y_line,
):
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(train_X_line, train_y_line)
    preds_base = baseline.predict(test_X_line)
    print(f"MAE: {mean_absolute_error(test_y_line, preds_base)}")
    print(f"R2: {r2_score(test_y_line, preds_base)}")
    return


@app.cell
def _(plt, preds_line, sns, test_y_line):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=test_y_line, 
                    y=preds_line)
    plt.xlabel("Настоящая популярность")
    plt.ylabel('Предсказенная популярность')
    plt.plot([0,100], [0,100], "r--")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### R2-SCORE и График

    Мы наблюдаем, что показатель r2 очень маленький, что говорит нам о том, что между аудио-признаками и популярностью трека, очень маленькая корреляция. Поэтому стоит использовать другие признаки, такие как артиста или жанр треков. Но изначально их, нужно превратить в категориальные данные. Погрешность модели примерно в районе 15, что говорит нам о том, что модель почти не даёт улучшения по сравнению с базовой моделью. Также мы наблюдаем, как признаки влияют на популярность, но у них разный числовой диапазон, что можно исправить добавлением z-score.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Метод ближайших соседей.

    Добавляем StandardScaler для того, чтобы стандатизировать данные из разных признаков, по той причине, что в разных признаках может быть числовой диапазон разный.
    """)
    return


@app.cell
def _(StandardScaler, train_X_1, val_X_1):
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X_1)
    val_X_scaled = scaler.transform(val_X_1)
    return train_X_scaled, val_X_scaled


@app.cell
def _(
    KNeighborsRegressor,
    mean_absolute_error,
    train_X_scaled,
    train_y_1,
    val_X_scaled,
    val_y_1,
):
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(train_X_scaled, train_y_1)
    knn_preds = knn_model.predict(val_X_scaled)
    print(mean_absolute_error(val_y_1, knn_preds))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Подсчитываем самое оптимальное количество соседей

    Мы видим, что с возрастанием количество соседей для регрессивного метода ближайших соседей увеличивается, также и погрешность модели. С чем это связано пока неизвестно
    """)
    return


@app.cell
def _(
    KNeighborsRegressor,
    mean_absolute_error,
    train_X_scaled,
    train_y_1,
    val_X_scaled,
    val_y_1,
):
    neighbors = [1, 2, 3, 4, 5, 7, 10, 20, 50, 100]
    for neighbor in neighbors:
        knn_model_1 = KNeighborsRegressor(n_neighbors=neighbor)
        knn_model_1.fit(train_X_scaled, train_y_1)
        knn_preds_1 = knn_model_1.predict(val_X_scaled)
        mae = mean_absolute_error(val_y_1, knn_preds_1)
        print(f"Соседи: {neighbor} \t Погрешность: {mae:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Мы наблюдаем, что метод ближайших соседей не имеет сильного влияния по сравнению с линейной ригресси.

    С ростом числа соседей погрешность модели снижается — усреднение по большему числу соседей уменьшает влияние шума отдельных наблюдений (снижение дисперсии). Оптимальным оказалось k=50 (MAE≈14.91). При дальнейшем росте k модель начинает недообучаться — усреднение по слишком большому числу соседей размывает локальные закономерности данных.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Анасамбли. Рандомный лес и Дерево решений

    Контекст: Берем те же признаки, что и прошлый раз, чтобы посмотреть, какой будет погрешность модели теперь
    """)
    return


@app.cell
def _(base_features, spotify, target):
    y_1 = spotify[target]
    X_1 = spotify[base_features]
    print(X_1.shape, y_1.shape)
    X_1.head()
    return X_1, y_1


@app.cell
def _(X_1, random_state, train_test_split, y_1):
    train_X_1, val_X_1, train_y_1, val_y_1 = train_test_split(X_1, y_1, random_state=random_state)
    return train_X_1, train_y_1, val_X_1, val_y_1


@app.cell
def _(mo):
    mo.md(r"""
    ### Подбор глубины дерева решений (число листьев)
    """)
    return


@app.cell
def _(
    DecisionTreeRegressor,
    mean_absolute_error,
    train_X_1,
    train_y_1,
    val_X_1,
    val_y_1,
):
    candidate_max_leaf_nodes = [5, 10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000]

    def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
        spot_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        spot_model.fit(train_X, train_y)
        val_predictions = spot_model.predict(val_X)
        mae = mean_absolute_error(val_y, val_predictions)
        return mae

    for max_leaf_nodes in candidate_max_leaf_nodes:
        mae = get_mae(max_leaf_nodes, train_X_1, val_X_1, train_y_1, val_y_1)
        print("Листья: %d \t\t Погрешность: %d" % (max_leaf_nodes, mae))
    return


@app.cell
def _(
    DecisionTreeRegressor,
    mean_absolute_error,
    random_state,
    train_X_1,
    train_y_1,
    val_X_1,
    val_y_1,
):
    spot_model_default = DecisionTreeRegressor(max_leaf_nodes=5, random_state=random_state)
    spot_model_default.fit(train_X_1, train_y_1)
    val_predictions_default = spot_model_default.predict(val_X_1)
    print("Плохая погрешность: {:,.2f}".format(mean_absolute_error(val_predictions_default, val_y_1)))
    return


@app.cell
def _(
    DecisionTreeRegressor,
    mean_absolute_error,
    random_state,
    train_X_1,
    train_y_1,
    val_X_1,
    val_y_1,
):
    spot_model_500 = DecisionTreeRegressor(max_leaf_nodes=500, random_state=random_state)
    spot_model_500.fit(train_X_1, train_y_1)
    val_predictions_500 = spot_model_500.predict(val_X_1)
    print("Лучшая погрешность: ", (mean_absolute_error(val_predictions_500, val_y_1)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Обучая модель на признаках `danceability`, `energy`, `acousticness`,
    `valence`, `tempo`, `loudness`, `speechiness`:

    Дерево с 500 листьями показало наименьший MAE, с 5 листьями — самый
    большой (недообучение). Но погрешность всё ещё слишком высока, чтобы
    считать модель точной — поэтому дальше пробуем случайный лес вместо
    одного дерева. Также мы видим, что после 1500 листьев идет переобучение модели. Поэтому самым оптимальным для нас вариантом остается дерево с 500 листьями. Решающие деревья не показали столь сильного роста, сравнивая с методом ближайших соседей.
    """)
    return


@app.cell
def _(
    RandomForestRegressor,
    mean_absolute_error,
    n_jobs,
    random_state,
    rf_n_estimators,
    train_X_1,
    train_y_1,
    val_X_1,
    val_y_1,
):
    rf_model_0 = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=random_state, n_jobs=n_jobs)
    rf_model_0.fit(train_X_1, train_y_1)
    rf_prediction_0 = rf_model_0.predict(val_X_1)
    print(mean_absolute_error(val_y_1, rf_prediction_0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Случайный лес дал MAE примерно такой же результат. Во всех моделях коррелирует в основном одно число в диапазоне от 14 до 16, не давая никакой прироста с базовой моделью.
    Дальше пробуем добавить ещё один признак, напрямую коррелирующий с
    треками — жанр (`genre`).

    ## Модель №2: добавляем жанр
    """)
    return


@app.cell
def _(RandomForestRegressor, features_with_genre, mean_absolute_error, n_jobs, random_state, rf_n_estimators, spotify, target):
    y_2 = spotify[target]
    X_2 = spotify[features_with_genre].copy()
    X_2['genre'] = X_2['genre'].astype(object)
    numerical_cols = [col for col in X_2.columns if X_2[col].dtype in ['float64', 'int64']]
    categorical_cols = [col for col in X_2.columns if X_2[col].dtype == 'object']

    def score_dataset(train_X, val_X, train_y, val_y):
        rf_model = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=random_state, n_jobs=n_jobs)
        rf_model.fit(train_X, train_y)
        rf_prediction = rf_model.predict(val_X)
        return mean_absolute_error(val_y, rf_prediction)

    return X_2, categorical_cols, numerical_cols, score_dataset, y_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Кодирование категориального признака `genre`

    Использование категорильного признака с ordinal возможна только на решающих деревьях и рандомном лесе, так как в остальных моделях такая кодировка не подходит.
    """)
    return


@app.cell
def _(X_2, random_state, train_test_split, y_2):
    train_X_2, val_X_2, train_y_2, val_y_2 = train_test_split(X_2, y_2, random_state=random_state)

    object_cols = [col for col in train_X_2.columns if train_X_2[col].dtype == "object"]
    good_label = [col for col in object_cols if set(val_X_2[col]).issubset(set(train_X_2[col]))]
    bad_label = list(set(object_cols) - set(good_label))
    print("Категориальные столбцы, которые будут ordinal", good_label)
    print("Категориальные столбцы, которые будут удалены с датасета", bad_label)
    return good_label, train_X_2, train_y_2, val_X_2, val_y_2


@app.cell
def _(good_label, train_X_2, val_X_2):
    from sklearn.preprocessing import OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    train_X_2[good_label] = ordinal_encoder.fit_transform(train_X_2[good_label])
    val_X_2[good_label] = ordinal_encoder.transform(val_X_2[good_label])
    return (OrdinalEncoder,)


@app.cell
def _(score_dataset, train_X_2, train_y_2, val_X_2, val_y_2):
    columns_null = [col for col in train_X_2.columns if train_X_2[col].isnull().any()]
    reduced_X_train = train_X_2.drop(columns_null, axis=1)
    reduced_X_valid = val_X_2.drop(columns_null, axis=1)

    print(score_dataset(reduced_X_train, reduced_X_valid, train_y_2, val_y_2))
    return


@app.cell
def _(pd, score_dataset, train_X_2, train_y_2, val_X_2, val_y_2):
    from sklearn.impute import SimpleImputer

    col_imp = SimpleImputer()
    imputed_train_X = pd.DataFrame(col_imp.fit_transform(train_X_2))
    imputed_val_X = pd.DataFrame(col_imp.transform(val_X_2))

    imputed_train_X.columns = train_X_2.columns
    imputed_val_X.columns = val_X_2.columns

    print(score_dataset(imputed_train_X, imputed_val_X, train_y_2, val_y_2))
    return (SimpleImputer,)


@app.cell
def _(
    SimpleImputer,
    pd,
    score_dataset,
    train_X_2,
    train_y_2,
    val_X_2,
    val_y_2,
):
    final_train_X = train_X_2.copy()
    final_val_X = val_X_2.copy()

    for col in train_X_2.columns:
        final_train_X[col + " _Найдено или нет"] = final_train_X[col].isnull()
        final_val_X[col + " _Найдено или нет"] = final_val_X[col].isnull()

    my_imp = SimpleImputer()
    final_train_X = pd.DataFrame(my_imp.fit_transform(final_train_X))
    final_val_X = pd.DataFrame(my_imp.transform(final_val_X))

    print(score_dataset(final_train_X, final_val_X, train_y_2, val_y_2))
    return


@app.cell
def _(train_X_2):
    print(train_X_2.isnull().sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Добавлен новый признак — жанры треков. Так как это был текстовый
    (object) признак, превратили его в категориальный через
    `OrdinalEncoder`. С random forest (50 деревьев) MAE получился **~11.18**
    — это меньше на 4 пункта с простым рандомным лесом, что показывает о том, что корреляция жанром трека и его популярностью весьма сильная. Все три способа чистки пропусков дали
    одинаковый результат, потому что пропусков в этих строках на самом деле
    не было — что подтверждает последняя проверка `isnull().sum()`.

    ### Собираем всё в один pipeline
    """)
    return


@app.cell
def _(
    OrdinalEncoder,
    RandomForestRegressor,
    SimpleImputer,
    categorical_cols,
    mean_absolute_error,
    n_jobs,
    numerical_cols,
    rf_n_estimators,
    train_X_2,
    train_y_2,
    val_X_2,
    val_y_2,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    numerical_transformer = SimpleImputer(strategy='constant')

    categorial_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal_encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorial_transformer, categorical_cols),
    ])
    rf_model_2 = RandomForestRegressor(n_estimators=rf_n_estimators, n_jobs=n_jobs, random_state=0)

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', rf_model_2),
    ])
    clf.fit(train_X_2, train_y_2)
    preds = clf.predict(val_X_2)

    print(mean_absolute_error(val_y_2, preds))
    return (clf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Пайплайн использован для ускорения препроцессинга. Результат почти не
    изменился, MAE остался в том же диапазоне.

    ### Кросс-валидация
    """)
    return


@app.cell
def _(X_2, clf, y_2):
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = -1 * cross_val_score(clf, X_2, y_2, cv=cv, scoring="neg_mean_absolute_error")

    print("Среднее значение MAE: ", scores.mean())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Кросс-валидация не улучшила результат, но позволяет нам удостоверится в правдивости данных, так как проверка была по группам.

    ### XGBoost
    """)
    return


@app.cell
def _(mean_absolute_error, train_X_2, train_y_2, val_X_2, val_y_2):
    from xgboost import XGBRegressor

    my_model_1 = XGBRegressor(random_state=0)
    my_model_1.fit(train_X_2, train_y_2)

    predictions_1 = my_model_1.predict(val_X_2)
    print(mean_absolute_error(val_y_2, predictions_1))
    return (XGBRegressor,)


@app.cell
def _(
    XGBRegressor,
    mean_absolute_error,
    train_X_2,
    train_y_2,
    val_X_2,
    val_y_2,
):
    my_model_2 = XGBRegressor(n_estimators=100, learning_rate=0.05)
    my_model_2.fit(train_X_2, train_y_2)

    predictions_2 = my_model_2.predict(val_X_2)
    print(mean_absolute_error(val_y_2, predictions_2))
    return


@app.cell
def _(
    XGBRegressor,
    mean_absolute_error,
    train_X_2,
    train_y_2,
    val_X_2,
    val_y_2,
):
    my_model_3 = XGBRegressor(n_estimators=1000, learning_rate=0.03)
    my_model_3.fit(train_X_2, train_y_2)

    predictions_3 = my_model_3.predict(val_X_2)
    print(mean_absolute_error(val_y_2, predictions_3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Общая статистика популярности

    - среднее значение: 33.24
    - минимальное значение: 0
    - максимальное значение: 100
    - самый популярный трек — "Unholy (feat. Kim Petras)"
    - самый длинный трек — "Unity (Voyage Mix) Pt. 1"

    Средняя популярность треков ~33 — большинство треков не хиты, есть
    большая разница между популярными и нишевыми треками, жанры сильно
    различаются по энергетике и настроению.
    """)
    return


@app.cell
def _(ms_per_minute, spotify):
    spotify["duration_min"] = spotify.duration_ms / ms_per_minute
    most_duration_id = spotify.duration_min.idxmax()
    most_duration = spotify['song'][most_duration_id]
    most_duration
    return


@app.cell
def _(np, spotify):
    np_popularity = spotify.popularity.to_numpy()

    stats = {
        'mean': np.mean(np_popularity),
        'median': np.median(np_popularity),
        'std': np.std(np_popularity),
        'percentiles': np.percentile(np_popularity, [25, 50, 75]),
    }
    stats
    return


@app.cell
def _(plt, sns, spotify):
    fig_reg, ax_reg = plt.subplots(figsize=(8, 4))
    sns.regplot(data=spotify, x='popularity', y='danceability', ax=ax_reg)
    fig_reg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Жанры: energy / danceability / valence

    Больше всего треков — в самых популярных жанрах (распределение
    неравномерное между категориями). Самый "позитивный" жанр (по
    `valence`) — `forro`, самый "энергичный" (по `energy`) — `happy`.
    """)
    return


@app.cell
def _(spotify):
    genre_stats = spotify.groupby("genre")[['energy', 'danceability', 'valence', "acousticness"]].mean()

    top_genres = spotify.genre.value_counts().head(20).index
    genre_stats = genre_stats.loc[top_genres]
    return (genre_stats,)


@app.cell
def _(genre_stats, plt, sns):
    fig_heat, ax_heat = plt.subplots(figsize=(8, 5))
    sns.heatmap(genre_stats, annot=True, ax=ax_heat)
    fig_heat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Настроение трека (`mood`)

    Новый столбец `mood` определяет настроение трека по `valence`:
    `Happy` (> 0.7), `Sad` (< 0.3), `Neutral` (остальное). Показаны 2
    подхода — через `apply()` и через векторный `np.select` (последний
    быстрее на больших данных).
    """)
    return


@app.cell
def _(mood_happy, mood_sad, spotify):
    def mood_valence(row):
        if row.valence > mood_happy:
            return 'Happy'
        if row.valence < mood_sad:
            return 'Sad'
        else:
            return 'Neutral'

    spotify['mood'] = spotify.apply(mood_valence, axis=1)

    mood_count = spotify.groupby('mood').size()
    mood_count
    return


@app.cell
def _(mood_happy, mood_sad, np, spotify):
    # Векторный вариант (быстрее apply на больших данных):
    conditions_mood = [
        (spotify['valence'] > mood_happy),
        (spotify['valence'] < mood_sad),
        ((spotify['valence'] > mood_sad) & (spotify['valence'] < mood_happy)),
    ]
    choices_mood = ["Happy", "Sad", "Neutral"]

    new_mood = np.select(conditions_mood, choices_mood, default='unknown')
    spotify['new_mood'] = new_mood
    spotify['new_mood']
    return


@app.cell
def _(plt, sns, spotify):
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=spotify, x='valence', y='energy', hue='new_mood', ax=ax_scatter)
    fig_scatter
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Топ-10 исполнителей по числу треков

    The Beatles (279), George Jones (271), Stevie Wonder (236), Linkin
    Park (224), Ella Fitzgerald (222), Prateek Kuhad (217), Feid (202),
    Chuck Berry (190), Håkan Hellström (183), OneRepublic (181).

    Дальше для этих исполнителей считаем среднюю популярность их треков —
    так видно не только частоту появления в датасете, но и кто из них в
    среднем успешнее с точки зрения стриминга.
    """)
    return


@app.cell
def _(spotify):
    top_artists = spotify.groupby('artist').size().sort_values(ascending=False).head(10)
    top_10_names = top_artists.index
    total = spotify[spotify['artist'].isin(top_10_names)].groupby('artist').popularity.mean()
    return (total,)


@app.cell
def _(plt, sns, total):
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    sns.barplot(x=total.index, y=total.values, ax=ax_bar)
    plt.setp(ax_bar.get_xticklabels(), rotation=45)
    fig_bar
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Финальный вид датасета

    Переименование колонок уже сделали в самом начале (`song`, `artist`,
    `genre`), поэтому здесь просто оставляем ключевые признаки и
    сортируем по популярности.
    """)
    return


@app.cell
def _(spotify):
    new_spotify = spotify[['song', 'artist', 'genre', 'popularity', 'duration_min', 'mood']]
    final_df = new_spotify.sort_values('popularity', ascending=False)
    final_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Матрица числовых признаков

    `mtx_all` — только числовые признаки, годные для дальнейшего ML:
    `danceability`, `energy`, `acousticness`, `valence`, `tempo`.
    """)
    return


@app.cell
def _(spotify):
    mtx_all = spotify[['danceability', 'energy', 'acousticness', 'valence', 'tempo']].to_numpy()
    print(mtx_all.shape)
    print(mtx_all.mean(axis=0))
    print(mtx_all.min(axis=0))
    print(mtx_all.max(axis=0))
    return


@app.cell
def _(plt, sns, spotify):
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    sns.histplot(data=spotify, x='popularity', kde=True, ax=ax_hist)
    ax_hist.axvline(spotify['popularity'].mean(), ls='--', label='mean')
    ax_hist.axvline(spotify['popularity'].median(), ls=':', label='median')
    ax_hist.legend()
    ax_hist.set_title('Popularity all track')
    fig_hist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Большинство треков имеют низкую популярность, распределение с
    выраженным правым скосом — высокопопулярные треки редки, очень
    небольшое число достигает максимума.
    """)
    return


@app.cell
def _(plt, sns, spotify):
    fig_kde, ax_kde = plt.subplots()
    sns.kdeplot(data=spotify, x='energy', fill=True, label='energy', ax=ax_kde)
    sns.kdeplot(data=spotify, x='danceability', fill=True, label='danceability', ax=ax_kde)
    sns.kdeplot(data=spotify, x='valence', fill=True, label='valence', ax=ax_kde)
    ax_kde.legend()
    fig_kde
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `energy` — небольшая асимметрия распределения, `danceability` —
    умеренный скос, `valence` — наиболее близко к симметричному среди
    трёх признаков.

    ## Hit / Non-Hit и нормализация

    Новый признак `popular_hit`: `Hit`, если `popularity >= 70`, иначе
    `Non-Hit`. Дальше строим матрицу из `energy`, `danceability`,
    `valence` и стандартизируем её (Z-score: вычесть среднее, поделить на
    стандартное отклонение).
    """)
    return


@app.cell
def _(hit_threshold, np, spotify, target):
    conditions_hit = np.array([
        (spotify[target] >= hit_threshold),
        (spotify[target] < hit_threshold),
    ])
    choices_hit = np.array(['Hit', 'Non-Hit'])

    spotify['popular_hit'] = np.select(conditions_hit, choices_hit, default='unknown')

    spotify.groupby('popular_hit')[['energy', 'danceability', 'valence']].mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Вывод:**

    - Разделение на Hit / Non-Hit показывает различия в характеристиках
      популярных и менее популярных треков, а нормализация признаков
      делает дальнейший анализ и возможное ML более корректным.
    - Популярные треки имеют чуть более высокие `energy` и `valence`.
    - Некоторые жанры стабильно более "позитивные".
    - Большинство треков имеют среднюю популярность → рынок не перегрет
      хитами.

    Данные преобразованы в числовую матрицу признаков, что делает
    возможным дальнейшее применение методов машинного обучения.
    """)
    return


if __name__ == "__main__":
    app.run()
