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
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score

    return (
        DecisionTreeRegressor,
        RandomForestRegressor,
        mean_absolute_error,
        np,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(pd):
    spotify_raw = pd.read_csv(r"C:\Users\posion\marimo_notebooks\__datasets__\spotify_data.csv")
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
    - После этого удалили дубликаты по `track_id`

    В итоге количество строк уменьшилось со 114000 до 89741 — убрали
    повторяющиеся треки.

    Здесь же сразу переименовываю столбцы (`track_name → song`,
    `artists → artist`, `track_genre → genre`) — в исходном ноутбуке это
    делалось в самом конце, но по факту использовалось (`genre`, `artist`)
    значительно раньше, так что переносим один раз в начало, чтобы всё
    дальше было последовательно и ничего не падало по `KeyError`.

    **Вывод:** после очистки данные стали более надёжными для анализа и
    исключили повторяющиеся записи.
    """)
    return


@app.cell
def _(spotify_raw):
    missing_count = spotify_raw.isnull().sum()

    replacement_artist = spotify_raw.artists.fillna("Unknown Artist")
    replacement_album = spotify_raw.album_name.fillna("Unknown Album_name")

    spotify = spotify_raw.drop_duplicates(subset=["track_id"])
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
    ## Целевая переменная и признаки (модель №1, только числовые фичи)

    Контекст: EDA уже проведён, известно, какие признаки связаны с
    популярностью. Формально выделяем целевую переменную `y` (`popularity`)
    и матрицу признаков `X`, используя колонки: `danceability`, `energy`,
    `acousticness`, `valence`, `tempo`, `loudness`, `speechiness`.
    """)
    return


@app.cell
def _(spotify):
    features_1 = [
        'danceability', 'energy', 'acousticness', 'valence',
        'tempo', 'loudness', 'speechiness',
    ]
    y_1 = spotify["popularity"]
    X_1 = spotify[features_1]
    print(X_1.shape, y_1.shape)
    X_1.head()
    return X_1, y_1


@app.cell
def _(X_1, train_test_split, y_1):
    train_X_1, val_X_1, train_y_1, val_y_1 = train_test_split(X_1, y_1, random_state=1)
    return train_X_1, train_y_1, val_X_1, val_y_1


@app.cell(hide_code=True)
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
    candidatea_max_leaf_nodes = [5, 20, 50, 100, 500]

    def get_most_lead(max_leaf_nodes, train_X, val_X, train_y, val_y):
        spot_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        spot_model.fit(train_X, train_y)
        val_prediction = spot_model.predict(val_X)
        mae = mean_absolute_error(val_y, val_prediction)
        return mae

    for max_leaf_nodes in candidatea_max_leaf_nodes:
        most_mae = get_most_lead(max_leaf_nodes, train_X_1, val_X_1, train_y_1, val_y_1)
        print("Листья: %d \t\t Погрешность: %d" % (max_leaf_nodes, most_mae))
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
    spot_model_default = DecisionTreeRegressor(random_state=1)
    spot_model_default.fit(train_X_1, train_y_1)
    val_prediction_default = spot_model_default.predict(val_X_1)
    print("Плохая погрешность: {:,.0f}".format(mean_absolute_error(val_prediction_default, val_y_1)))
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
    spot_model_500 = DecisionTreeRegressor(max_leaf_nodes=500, random_state=1)
    spot_model_500.fit(train_X_1, train_y_1)
    val_prediction_500 = spot_model_500.predict(val_X_1)
    print("Лучшая погрешность: ", (mean_absolute_error(val_prediction_500, val_y_1)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Обучая модель на признаках `danceability`, `energy`, `acousticness`,
    `valence`, `tempo`, `loudness`, `speechiness`:

    Дерево с 500 листьями показало наименьший MAE, с 5 листьями — самый
    большой (недообучение). Но погрешность всё ещё слишком высока, чтобы
    считать модель точной — поэтому дальше пробуем случайный лес вместо
    одного дерева.
    """)
    return


@app.cell
def _(
    RandomForestRegressor,
    mean_absolute_error,
    train_X_1,
    train_y_1,
    val_X_1,
    val_y_1,
):
    rf_model_0 = RandomForestRegressor(n_estimators=50, random_state=1, n_jobs=-1)
    rf_model_0.fit(train_X_1, train_y_1)
    rf_predication_0 = rf_model_0.predict(val_X_1)
    print(mean_absolute_error(val_y_1, rf_predication_0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Случайный лес дал MAE ниже примерно на 2 пункта — хороший результат.
    Дальше пробуем добавить ещё один признак, напрямую коррелирующий с
    треками — жанр (`genre`).

    ## Модель №2: добавляем жанр
    """)
    return


@app.cell
def _(RandomForestRegressor, mean_absolute_error, spotify):
    # cell-16
    features_2 = [
        'danceability', 'energy', 'acousticness', 'valence',
        'tempo', 'loudness', 'speechiness', 'genre',
    ]
    y_2 = spotify["popularity"]
    X_2 = spotify[features_2].copy()
    X_2['genre'] = X_2['genre'].astype(object)
    numerical_cols = [col for col in X_2.columns if X_2[col].dtype in ['float64', 'int64']]
    categorial_cols = [col for col in X_2.columns if X_2[col].dtype == 'object']

    def score_dataset(train_X, val_X, train_y, val_y):
        rf_model = RandomForestRegressor(n_estimators=50, random_state=1, n_jobs=-1)
        rf_model.fit(train_X, train_y)
        rf_predication = rf_model.predict(val_X)
        return mean_absolute_error(val_y, rf_predication)

    return X_2, categorial_cols, numerical_cols, score_dataset, y_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Кодирование категориального признака `genre`
    """)
    return


@app.cell
def _(X_2, train_test_split, y_2):
    train_X_2, val_X_2, train_y_2, val_y_2 = train_test_split(X_2, y_2, random_state=1)

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
    `OrdinalEncoder`. С random forest (50 деревьев) MAE получился **~10.3**
    — заметно точнее, чем без жанра. Все три способа чистки пропусков дали
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
    categorial_cols,
    mean_absolute_error,
    numerical_cols,
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
        ('cat', categorial_transformer, categorial_cols),
    ])
    rf_model_2 = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=0)

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
    Кросс-валидация не улучшила результат — дальше используем обычную
    валидацию, кросс-валидация лишь отнимает время.

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
def _(spotify):
    population_set_mean = spotify.popularity.mean()
    population_set_max = spotify.popularity.max()
    population_set_min = spotify.popularity.min()

    most_popular_idmax = spotify.popularity.idxmax()
    most_popular = spotify['song'][most_popular_idmax]

    spotify["duration_min"] = spotify.duration_ms / 60000
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

    genre_positiv = genre_stats['valence'].idxmax()
    genre_energy = genre_stats['energy'].idxmax()
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
def _(spotify):
    def Mood_Valence(row):
        if row.valence > 0.7:
            return 'Happy'
        if row.valence < 0.3:
            return 'Sad'
        else:
            return 'Neutral'

    spotify['mood'] = spotify.apply(Mood_Valence, axis='columns')

    mood_count = spotify.groupby('mood').value_counts()
    mood_count
    return


@app.cell
def _(np, spotify):
    # Векторный вариант (быстрее apply на больших данных):
    conditions_mood = [
        (spotify['valence'] > 0.7),
        (spotify['valence'] < 0.3),
        ((spotify['valence'] > 0.3) & (spotify['valence'] < 0.7)),
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
def _(np, spotify):
    conditions_hit = np.array([
        (spotify['popularity'] >= 70),
        (spotify['popularity'] < 70),
    ])
    choices_hit = np.array(['Hit', 'Non-Hit'])

    spotify['popular_hit'] = np.select(conditions_hit, choices_hit, default='unknown')

    mtx = spotify[['energy', 'danceability', 'valence']].to_numpy()
    mean = mtx.mean(axis=0)
    std = mtx.std(axis=0)

    standardized_data = (mtx - mean) / std
    spotify.groupby('popular_hit')[['energy', 'danceability', 'valence']].mean()
    return


@app.cell
def _(spotify):
    spotify['popular_hit'].value_counts()
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
