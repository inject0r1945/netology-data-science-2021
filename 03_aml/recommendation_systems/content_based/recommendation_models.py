import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import os

class ContentBasedMovieRecommendations:
    """ Класс для построения рекомендаций пользователю на основе датасета MovieLens
    
    Параметры
    ---------
    user_id : int или str
      Идентификатор пользователя, для которого будем строить модель. Принимает значение int или 'auto'. D случае 
      auto пользователь будет выбран автоматически по критерию максимального количества оценок к фильмам
    movies_csv : str
      Путь к файлу movies.csv из датасета MovieLens
    ratings_csv : str
      Путь к файлу ratings.csv из датасета MovieLens
    tags_csv : str
      Путь к файлу tags.csv из датасета MovieLens
    svr_C : float
      Коэффициент регуляризации для регрессора SVR, смотри справку по sklearn.svm.SVR
    svr_tol : float
       Минимальный допуск изменения качества SVR до остановки обучения модели, смотри справку по sklearn.svm.SVR
    svr_kernel : str
      Тип ядра для регрессора SVR, смотри справку по sklearn.svm.SVR
    svr_deegre : str
      Степень полинома для ядра poly, смотри справку по sklearn.svm.SVR
    recommend_count : int
      Количество фильмов для итоговой рекомендации
    random_recommend_films : int
      Количество случайных фильмов с высоким общим рейтингом, которые будут подмешиваться в рекомендации
    n_neighbors : int
      Количество сходных по жанру и тегам фильмов для выдачи рекомендаций. При получении рекомендации
      модель выберет указанное количество фильмов, предскажет для них оценку пользователя и выберет топ фильмов 
      уровню оценки. размер топа определяется в recommend_count.
    """
    
    def __init__(self, user_id, movies_csv, ratings_csv, tags_csv, 
                 svr_C=1.0, svr_tol=1e-3, recommend_count=5, random_recommend_films=1,
                 svr_kernel='rbf', svr_gamma='scale', svr_degree=3, n_neighbors=100):
        
        self._movies_csv = movies_csv
        self._ratings_csv = ratings_csv
        self._tags_csv = tags_csv
        self._user_id = int(user_id) if user_id != 'auto' else user_id
        
        self._svr_C = svr_C
        self._svr_tol = svr_tol
        self._svr_kernel = svr_kernel
        self._svr_gamma = svr_gamma
        self._svr_degree = svr_degree
        self._random_recommend_films = 1
        self._recommend_count = recommend_count
        self._n_neighbors = n_neighbors
        
        self.__init_ml_models()
        
        # Флаг подготовки датафреймов к обучению
        self.__df_is_prepared = False    
        
        
    def __init_ml_models(self):
        
        self._tfidf = TfidfVectorizer()
        
        self._predict_rating_model = SVR(C=self._svr_C, tol=self._svr_tol, kernel=self._svr_kernel, gamma=self._svr_gamma,
                                         degree=self._svr_degree)   
        
        self._neighbors_model = NearestNeighbors(n_neighbors=self._n_neighbors, n_jobs=-1, metric='euclidean') 
        
        self._scaler_tfidf = StandardScaler()
        self._scaler_regression = StandardScaler()        
        
        self.__model_is_fited = False   
        
        
    @property
    def svr_C(self, svr_C):
        return self._svr_C
    
    @svr_C.setter
    def svr_C(self, svr_C):
        self._svr_C = svr_C
        self.__init_ml_models()
        
    @property
    def svr_tol(self, svr_tol):
        return self._svr_tol
    
    @svr_tol.setter
    def svr_tol(self, svr_tol):
        self._svr_tol = svr_tol
        self.__init_ml_models()
        
    @property
    def svr_kernel(self, svr_kernel):
        return self._svr_kernel
    
    @svr_kernel.setter
    def svr_kernel(self, svr_kernel):
        self._svr_kernel = svr_kernel
        self.__init_ml_models()  
        
    @property
    def svr_degree(self, svr_degree):
        return self._svr_degree
    
    @svr_degree.setter
    def svr_degree(self, svr_degree):
        self._svr_degree = svr_degree
        self.__init_ml_models()   
        
    @property
    def svr_gamma(self, svr_gamma):
        return self._svr_gamma
    
    @svr_gamma.setter
    def svr_gamma(self, svr_gamma):
        self._svr_gamma = svr_gamma
        self.__init_ml_models() 
        
    def prepare_dataframes(self):
        """ Подготовка датафреймов для обучения
        
        Результатом работы функции является датафрейм self._df_movies
        со всеми необходимыми данными для обучения моделей
        """
        
        # Читаем датафреймы из файлов CSV
        self._df_ratings = pd.read_csv(self._ratings_csv)
        self._df_tags = pd.read_csv(self._tags_csv)
        self._df_movies = pd.read_csv(self._movies_csv)
        
        if self._user_id != 'auto' and self._user_id not in self._df_ratings.userId.values:
            raise ValueError(f'Указанный пользователь с идентификатором {self._user_id} отсутствует в базе с рейтингами, требуется выбрать другого пользователя')
        
        if self._user_id == 'auto':
            self._user_id = self._df_ratings.groupby('userId', axis=0).agg({'rating': 'count'}).sort_values('rating', ascending=False).index[0]
        
        # df_movies является основным датафреймом, добавляем в него дополнительные данные
        # в виде средней и медианной оценки всех пользователей  каждого фильма, а так же СКО эти оценок
        self._df_movies.loc[:, 'rating_mean'] = self._df_movies.movieId.apply(lambda x: self._df_ratings[self._df_ratings.movieId == x]['rating'].mean())
        self._df_movies.loc[:, 'rating_median'] = self._df_movies.movieId.apply(lambda x: self._df_ratings[self._df_ratings.movieId == x]['rating'].median())
        self._df_movies.loc[:, 'rating_std'] = self._df_movies.movieId.apply(lambda x: self._df_ratings[self._df_ratings.movieId == x]['rating'].std())
        
        # Если средний рейтинг существует, то пропущенные значения rating_std заполняем 0
        # так как это говорит о том, что оценка для фильма одинаковая у всех пользователей
        self._df_movies.loc[(~self._df_movies.rating_mean.isna()) & 
                            (self._df_movies.rating_std.isna()), 'rating_std'] = 0
        
        # Удаляем фильмы, у которых вообще нет никаких оценок
        self._df_movies.dropna(subset=['rating_mean', 'rating_median', 'rating_std'], inplace=True)
        
        # Теперь к df_movies прикрепим оценки нашего выбранного пользователя для каждого фильма, если они есть
        self._df_movies = pd.merge(self._df_movies, 
                                   self._df_ratings.loc[self._df_ratings.userId == self._user_id, 
                                                        ['movieId', 'rating']],
                                   on='movieId', how='left')
        
        self._df_movies.rename(columns={'rating': 'user_rating'}, inplace=True)
        
        # Находим идентификаторы фильмов в оценках и тэгах, которые принадлежат выбранному пользователю
        user_movie_ids = list(set(self._df_ratings[self._df_ratings.userId == self._user_id].movieId.values) 
                              | set(self._df_tags[self._df_tags.userId == self._user_id].movieId.values))
        
        # Подготавливаем столбец с жанрами фильмов
        self._df_movies.genres = self._df_movies.genres.apply(self._prepare_movie_genres)
        
        # Для каждого фильма добавляем столбцец user_tags и записываем все тэги выбранного пользователя 
        self._df_movies.loc[self._df_movies.movieId.isin(user_movie_ids), 'user_tags'] = self._df_movies[self._df_movies.movieId.isin(user_movie_ids)].movieId.apply(self._prepare_movie_tags, user_id=self._user_id)
        
        # Если выбранный пользователь не поставил для какого-либо фильма теги, 
        # то заполняем это поле текстом NO_USER_TAG
        self._df_movies.loc[self._df_movies.user_tags.isna(), 'user_tags'] = 'NO_USER_TAG'
        
        # Ставим флаг уведомления о том, что датафрейм self._df_movies готов для обучения
        self.__df_is_prepared = True
        
    
    def export_prepared_data(self, dirpath):
        """ Экспорт подготовленных данных для обучения
        
        При большом обхеме данных функция prepare_dataframes отрабатывает очень долго. 
        Можно сделать экапорт подготовленных данных и в дальнейшем делать их импорт без повторной подготовки
        через prepare_dataframes
        
        Параметры
        ---------
        dirpath : str
          Путь к папке с сохраняемыми данными
          
        Результат
        ---------
        Сохранение подготовленных для обучения данных в файле в формате csv c разделителем в виде запятой
        """
        
        dirpath = dirpath + os.sep if dirpath[-1] != os.sep else dirpath
        if not os.path.exists(dirpath):
            print(f"Папка {dirpath} не существует, не удалось сохранить данные")
            return 1
        
        self._df_movies.to_csv(dirpath + 'movies.csv', index=False, sep=',')
        return 0
        
    def import_prepared_data(self, dirpath):
        """ Импорт подготовленных данных для обучения
        
        Параметры
        ---------
        dirpath : str
          Путь к файлам, полученным из метода export_prepared_data (movies.csv)
        """
        
        dirpath = dirpath + os.sep if dirpath[-1] != os.sep else dirpath
        if not os.path.exists(dirpath):
            print(f"Папка {dirpath} не существует, не удалось загрузить данные")
            return 1
        
        self._df_movies = pd.read_csv(dirpath + 'movies.csv', delimiter=',')
        self.__df_is_prepared = True
       
    
    @staticmethod
    def _prepare_movie_genres(genre):
        """ Преобразовывает текстовую строку жанра в формате "Adventure|Children|Fantasy"
        в "adventure children fantasy"
        
        Параметры
        ---------
        genre : str
          Жанры фильмов в формате "Adventure|Children|Fantasy"
          
        Результат
        ---------
        genre_processed : str
          Обработанная строка с жанрами фильма
          
        """
        genre_processed = ' '.join(genre.replace('-', '').replace(' ', '').lower().split('|'))
        return genre_processed
    
    
    def _prepare_movie_tags(self, movie_id, user_id):
        """ Для каждого фильма по его идентификатору находятся все теги для нужного пользователя 
        и выдаются в виде единого текста
        
        Перед запуском функции должен быть сформирован датафрейм self._df_tags
        
        Параметры
        ---------
        movie_id : int
          Идентификатор фильма
        user_id : int
          Идентификатор пользователя
          
        Результат
        ---------
        user_tags : str
          Строка с тэгами фильма с movie_id для пользователя user_id
        """
        
        user_tags = self._df_tags[(self._df_tags.movieId == movie_id) &
                                  (self._df_tags.userId == user_id)].loc[:, 'tag'].values
        user_tags = [str(tag).replace(' ', '').lower() for tag in user_tags]
        user_tags = ' '.join(user_tags)
        
        if not user_tags:
            user_tags = None
        
        return user_tags
    
    def fit(self, test_size=0.3):
        """ Обучение модели рекомендательной системы
        
        Параметры
        ---------
        test_size : float
          Размер тестовой выборки для обучения регрессионной модели предсказания оценки пользователя. 
          Принимает значения от 0 до 1
        """
        
        if not self.__df_is_prepared:
            self.prepare_dataframes()
             
        """
        Блок обучения модели соседей
        
        Для обучения этой модели требуются данные по всем фильмам без учета оценки выбранного пользователя
        """
           
        # Создаем копию датафрейма, для которой будем выполнять все преобразования
        df_model = self._df_movies.copy()
        # Сохраняем таргет в отдельный датафрейм
        df_target = df_model[['user_rating']]
        df_means_data= df_model[['rating_mean', 'rating_median', 'rating_std']]
        
        # Оставляем только нужные столбцы для обучения
        df_model = df_model.loc[:, ['user_tags', 'genres']]
        
        # Объединяем поля user_tags и genres в одно текстовое поле для операции TF-IDF
        df_model['text'] = df_model['user_tags'] + ' ' + df_model['genres']
        # Делаем реиндекс столбцов
        df_model = df_model.reindex(columns=['text'])
        
        # Получаем TF-IDF для всех данных модели
        df_model_tfidf = self._tfidf.fit_transform(df_model['text']).todense()
        
        # Удаляем колонку text из-за ненадобности
        df_model = pd.DataFrame(df_model_tfidf, columns=self._tfidf.get_feature_names())
        
        # Стандартизуем данные
        df_model_std = self._scaler_tfidf.fit_transform(df_model)
        
        df_model_std = pd.DataFrame(df_model_std, columns=df_model.columns)
        
        # Создаем специальную копию датафрейма модели, в которой нет оценок нашего пользователя
        # Требуется для рекомендаций только тех фильмов, которые пользователь не смотрел вообще
        # Делаем через self, так как в предсказаниях соседа будем обращаться к этому датафрему по результатам предикта
        df_model_neighbors = df_model_std[df_target.user_rating.isna()]
        self._df_movies_neighbors = self._df_movies[df_target.user_rating.isna()]
        
        # Обучаем модель соседей на всех данных, кроме оценки выбранного пользователя
        # Учим модель соседей только на TF-IDF
        #self._neighbors_model.fit(df_model_neighbors)
        self._neighbors_model.fit(df_model_neighbors)
        
        """
        Блок обучения модели линейной регрессии
        """
        
        # Собираем латафрейм модели для регрессии, здесь df_model берется из предыдущего этапа без стандартизации
        # Стандартизация для этого блока будет выполнена отдельно
        df_model = pd.concat([df_model, df_means_data, df_target], axis=1)
        
        # Дропаем все строки, которые содержат хотя бы один NaN. В функции подготовки датафреймов мы делали
        # необходимые заполнения пропусков.
        # В данном случае дропнутся только те фильмы, которым выбранный пользователь не ставил оценку
        df_model.dropna(inplace=True)
        
        df_target = df_model['user_rating']
        df_X = df_model.loc[:, ~df_model.columns.isin(['user_rating'])]
        
        #Создаем новый порядковый индекс после операции дропа, так как порядок нарушен
        df_X.index = np.arange(len(df_X))
        
        # Разбиваем выбору на обучающую и тестовую   
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(df_X, df_target, 
                                                                                    test_size=test_size)
        # Стандартизуем данные
        self.X_train_ = self._scaler_regression.fit_transform(self.X_train_)
        self.X_test_ = self._scaler_regression.transform(self.X_test_)
        
        # Линейную регрессию обучаем только на тренировочных данных
        self._predict_rating_model.fit(self.X_train_, self.y_train_)
        
        self.__model_is_fited = True
        
    def evaluate(self):
        """ Оценка точности предсказания оценки пользователя с помощью линейной регрессии
        """
        if not self.__model_is_fited:
            return
        
        y_predict = self._predict_rating_model.predict(self.X_test_)
        
        mae = mean_absolute_error(self.y_test_, y_predict)
        rmse = np.sqrt(mean_squared_error(self.y_test_, y_predict))
        print("MAE:", mae)
        print("RMSE:", rmse)
        
    
    def get_recommendations(self, movie_id):
        """ Получение рекомендации фильма для выбранного пользователя с предполагаемой лучшей оценкой
        
        Параметры
        ---------
        movie_id : int
          Идентификатор фильма, который посмотрел пользователь и для которого строим рекомендацию
        """
        
        if not self.__model_is_fited:
            print('Модель не обучена')
            return
        
        if movie_id not in self._df_movies.movieId.values:
            print(f'Фильм с идентификатором {movie_id} не существует')
            return
        
        # Получаем датафрейм с просмотренным фильмом, для которого нужно сделать рекомендацию
        df_reviewed_film = self._df_movies.loc[self._df_movies.movieId == movie_id]
        print('Пользователь посмотрел следующий фильм:')
        display(df_reviewed_film)
        
        
        # Найдем самые похожие фильмы с помощью модели ближайших соседей
        reviewed_film_text = df_reviewed_film['genres'] + ' ' + df_reviewed_film['user_tags']
        reviewed_film_tfidf = self._tfidf.transform(reviewed_film_text).todense()
        
        """
        reviewed_film_vector = np.hstack([df_reviewed_film.loc[:, ['rating_mean', 
                                                                  'rating_median', 'rating_std']].values,
                                          reviewed_film_tfidf])
        """
        reviewed_film_tfidf_std = self._scaler_tfidf.transform(reviewed_film_tfidf)
        
        neighbors_film_result = self._neighbors_model.kneighbors(reviewed_film_tfidf_std, return_distance=True)
        
        neighbors_film_ids = neighbors_film_result[1][0]
        neighbors_film_dist = neighbors_film_result[0][0]
        
        # Получаем датафрейм с ближайшими по смыслу фильмами по их индексам
        df_neighbors_film = self._df_movies_neighbors.iloc[neighbors_film_ids].copy()
        df_neighbors_film.loc[:, 'distance'] = neighbors_film_dist
        
        # Удаляем просмотренный фильм из выборки соседних
        df_neighbors_film = df_neighbors_film[df_neighbors_film.movieId != movie_id]
        
        # Теперь предскажем оценку пользователя для каждого фильма
        # Добавим в выборку случайные фильмы с рейтингом выше 4.5, чтобы пользователь мог посмотреть что-то принципиально новое
    
        best_rating_films_ids = self._df_movies.loc[(self._df_movies.rating_median > 4) & 
                                                    (self._df_movies.rating_std > 0) &
                                                    (self._df_movies.user_rating.isna()) &
                                                    (~self._df_movies.movieId.isin(df_neighbors_film.movieId.values)), 'movieId'].values
        if self._random_recommend_films > len(best_rating_films_ids):
            self._random_recommend_films = len(best_rating_films_ids)
    
        choised_random_films_ids = np.random.choice(best_rating_films_ids, size=self._random_recommend_films) 
        df_choised_random_films = self._df_movies[self._df_movies.movieId.isin(choised_random_films_ids)].copy()
        df_choised_random_films['distance'] = None
        
        df_neighbors_film = pd.concat([df_neighbors_film, df_choised_random_films], axis=0)
        
        # Для преобразований модели создаем отдельный датафрейм
        df_model = df_neighbors_film.copy()
        df_model['text'] = df_model['genres'] + ' ' + df_model['user_tags']
        df_model = df_model.reindex(columns=['text', 'rating_mean', 'rating_median', 'rating_std'])
        df_model_tfidf = self._tfidf.transform(df_model['text']).todense()
        df_model.drop(columns=['text'], inplace=True)
        
        df_model.index = np.arange(len(df_model))
        df_model = pd.concat([pd.DataFrame(df_model_tfidf, columns=self._tfidf.get_feature_names()),
                              df_model], axis=1)
        
        model_std = self._scaler_regression.transform(df_model)
        
        predict_user_rating = self._predict_rating_model.predict(model_std)
        
        df_neighbors_film.loc[:, 'user_rating_predict'] = predict_user_rating
        
        df_neighbors_film = df_neighbors_film.sort_values('user_rating_predict', ascending=False)[:self._recommend_count]
        df_neighbors_film.drop(columns='user_rating', inplace=True)
        df_neighbors_film.index = np.arange(len(df_neighbors_film))
        
        print(f'ТОП-{self._recommend_count} рекомендованных фильмов для просмотра в порядке максимальной предсказанной оценки пользователя:')
        display(df_neighbors_film)
            
    def print_stat(self):
        """ Статистика выбранного пользователя по датафрейму
        """
        if self.__df_is_prepared:
            print(f'Идентификатор пользователя: {self._user_id}')
            print('Кол-во оценок пользователя к фильмам:', (~self._df_movies.user_rating.isna()).sum())
            print('Всего кол-во фильмов:', len(self._df_movies))
            
        if self.__model_is_fited:
            print(f'Кол-во фичей при обучении регрессионной модели: {self.X_train_.shape[1]}')
            print(f'Кол-во фильмов при обучении регрессионной модели: {self.X_train_.shape[0]}')