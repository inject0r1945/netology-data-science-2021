import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ts_smooth_plot(series, smooth_funcs=[]):
    """ Печать временного ряда и результатов функций сглаживания
    
    Параметры
    ---------
    series : pandas series
      Временной ряд
    smooth_funcs : list
      Список словарей функций, которые будут выполнять сглаживание данных
      [{'Название функции': {'func': объект функции', 'params': {словарь параметров}}}]
      
      Пример:
      smooth_funcs = {
            "Простое скользящее среднее": {
                'func': simple_moving_average,
                'params': {'n_interval': 4}
            },
            "Взвешенное скользящее среднее": {
                'func': weighted_moving_average,
                'params': {'n_interval': 4, 'weights': [1, 2, 4, 8]}
            }, 
        }  
    """
    
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    line_styles = [
        ['green', '-'], ['red', '--'], ['blue', '-.'], ['orange', ':'],
        ['red', '-'], ['blue', '--'], ['orange', '-.'], ['green', ':'],
        ['blue', '-'], ['orange', '--'], ['green', '-.'], ['red', ':'],
        ['orange', '-'], ['green', '--'], ['red', '-.'], ['blue', ':'],
    ]
    
    line_styles.reverse()
    
    with plt.style.context('bmh'):
        
        linecolor, linetype = line_styles.pop()
        plt.figure(figsize=(17,6))
        plt.plot(series, label='Исходные значения', linestyle=linetype, color='black')
        
        for func_name, func_data in smooth_funcs.items():
            
            smooth_series = func_data['func'](series, **func_data['params'])
            linecolor, linetype = line_styles.pop() if len(line_styles) > 0 else ('black', '-')
            if len(func_data['params']) > 0:
                params_str = ', '.join(f'{param}={value}' for param, value in func_data['params'].items())
                func_name += f' ({params_str})'
            plt.plot(smooth_series, label=func_name, linestyle=linetype, color=linecolor,)
        
        plt.legend()
        plt.show()


def simple_moving_average(series, n_interval=4):
    """ Простое скользящее среднее
    
    Параметры
    ---------
    series : pandas series
      Значения временного ряда
    n_interval : int
      Окно усреднения
      
    Результат
    ---------
      smooth series : pandas series
        Сглаженные значения временного ряда
    """
    
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    return series.rolling(n_interval).mean()


def weighted_moving_average(series, n_interval=4, weights=None):
    """ Взвешенное скользящее среднее
    
    Параметры
    ---------
    series : pandas series
      Значения временного ряда
    n_interval : int
      Окно усреднения
    weights : list
      Список весов. Размер должен совпадать с размером окна усреднения
      Если None, то веса генерируются по степени 2.
      
    Результат
    ---------
      smooth series : pandas series
        Сглаженные значения временного ряда
    """
    
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    if weights and not isinstance(weights, list) and not isinstance(weights, tuple):
        weights = list(weights)
    else:
        weights = [2**n for n in range(n_interval)]
        
    return series.rolling(n_interval).apply(lambda s: sum(s * weights) / sum(weights) )


def simple_exponential_smoothing(series, alpha=None, n_interval=4):
    """ Простое экспоненциальное сглаживание
    
    Параметры
    ---------
    series : pandas series
      Значения временного ряда
    n_interval : int
      Окно усреднения для задания певрого значения EMA(t-1)
    alpha : float
      Параметр alpha в формуле простого экспоненциального среднего 
      EMA(t) = alpha*y(t) + (1-alpha)*EMA(t-1)
      
    Результат
    ---------
      smooth series : pandas series
        Сглаженные значения временного ряда
    
    """
    
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
        
    if len(series) < n_interval:
        raise ValueError("Значение n_interval должно быть больше размера серии")
    
    if not alpha:
        alpha = 2 / (n_interval + 1)
    
    smooth_result = [None for i in range(n_interval)]
    smooth_result.append(alpha*series[n_interval] + (1-alpha)*series[:n_interval].mean())
    
    for index in range(n_interval+1, len(series)):
        smooth_value = alpha*series[index] + (1-alpha)*smooth_result[index-1]
        smooth_result.append(smooth_value)
        
    return smooth_result


class DoubleExponentialSmoothing:
    """ Двойное экспоненциальное сглаживание для адитивной схемы временного ряда
    
    Параметры
    ---------
    alpha : float
      Коэфициент уровня, от 0 до 1
    beta : float
      Коэфициент тренда, от 0 до 1
    """
    
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        
    def fit(self, series):
        """ Подгонка модели
        
        Параметры
        ---------
        alpha : float
          Коэфициент уровня, от 0 до 1
        beta : float
          Коэфициент тренда, от 0 до 1
          
        Результат
        ---------
        smooth : series
          Сглаженные данные
        """
        
        self.smooth_data_ = []
        self.__level, self.__trend = series[0], series[1] - series[0]
        
        for index in range(1, len(series)):
            
            self.__prev_level,  self.__prev_trend = self.__level, self.__trend
            self.__level = self.alpha * series[index] + (1 - self.alpha) * (self.__prev_level + self.__prev_trend)
            self.__trend = self.beta * (self.__level - self.__prev_level) + (1 - self.beta) * self.__prev_trend
            self.smooth_data_.append(self.__level + self.__trend)
            
        self.smooth_data_ = pd.Series(self.smooth_data_)
        
        return self.smooth_data_
        
    def predict(self, n_predicts):
        """ Предсказание данных. Модель умеет корректно предсказывать только одну следующую точку
        
        Параметры
        ---------
        n_predicts : int
          Кол-во желаемых предсказанных значений
        
        Результат
        ---------
        pкedict : series
          Предсказаныне данные вместе со сглаженными
        """
        
        if 'smooth_data_' not in self.__dir__():
            print('Требуется запуск подгонки модели')
            return None
        
        smooth_data = self.smooth_data_.copy()
            
        predict_result = []
        
        level, trend = self.__level, self.__trend 
        for n in range(1, n_predicts+1):
            prev_level,  prev_trend = level, trend
            level = self.alpha * smooth_data.values[-1] + (1 - self.alpha) * (prev_level + prev_trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * prev_trend
            predict_result.append(level + trend)
            
        predict_result = pd.Series(predict_result)
        
        return smooth_data.append(pd.Series(predict_result), ignore_index=True)


class TripleExponentialSmoothing:
    """ Сглаживание и предсказание методом Хольта-Винтерса
    
    ВАЖНО: Данные во временном ряду должны иметь полные сезоны вначале ряда и в конце
    
    Параметры
    ---------
    alpha : float
      Коэфициент уровня, от 0 до 1
    beta : float
      Коэфициент тренда, от 0 до 1
    gamma : float
      Коэфициент сезонности, от 0 до 1
    season_len : int
      Кол-во значений серии в рамках сезона. Если сезонность проявляется в квартале, а значения в серии помесячные,
      то season_len = 4
    model_type : str
      Тип схемы временного ряда: add (адитивный) или mul (мультипликативный)
    
    """
    
    def __init__(self, alpha, beta, gamma, season_len=12, model_type='add'):
        
        if not 0 <= alpha <= 1:
            raise ValueError("Параметр alpha должен принимать значения от 0 до 1")
        if not 0 <= beta <= 1:
            raise ValueError("Параметр beta должен принимать значения от 0 до 1")
        if not 0 <= gamma <= 1:
            raise ValueError("Параметр gamma должен принимать значения от 0 до 1")
        if not isinstance(season_len, int):
            raise ValueError("Параметр season_len должен быть целым числом") 
        if season_len < 1:
            raise ValueError("Параметр season_len должен быть больше 1")
        if str(model_type).lower() not in ('add', 'mul'):
            raise ValueError("Параметр model_type может принимать только значения add или mul")
        
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._season_len = season_len
        self._model_type = str(model_type).lower()
        
       
    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, alpha):
        if not 0 <= alpha <= 1:
            raise ValueError("Параметр alpha должен принимать значения от 0 до 1")
        self._alpha = alpha
        
    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, beta):
        if not 0 <= beta <= 1:
            raise ValueError("Параметр beta должен принимать значения от 0 до 1")
        self._beta = beta
        
    @property
    def gamma(self):
        return self._gamma
    @gamma.setter
    def gamma(self, gamma):
        if not 0 <= gamma <= 1:
            raise ValueError("Параметр gamma должен принимать значения от 0 до 1")
        self._gamma = gamma
        
    @property
    def season_len(self):
        return self._season_len
    @season_len.setter
    def season_len(self, season_len):
        if not isinstance(season_len, int):
            raise ValueError("Параметр season_len должен быть целым числом") 
        if season_len < 1:
            raise ValueError("Параметр season_len должен быть больше 1")
        self._season_len = season_len
           
    @property
    def model_type(self):
        return self._model_type
    @model_type.setter
    def model_type(self, model_type):
        if str(model_type).lower() not in ('add', 'mul'):
            raise ValueError("Параметр model_type может принимать только значения add или mul")
        self._model_type = str(model_type).lower()
        
        
    def _init_trend_coef(self, series):
        """ Инициализация начального коэфициента тренда
        
        Параметры
        ---------
        series : pandas series
          Временной ряд
          
        Результат
        ---------
        trend_coef : float
          Начальное значение тренда
        """
        sum_val = 0.0
        for season_val_index in range(self._season_len):
            sum_val += float(series[season_val_index+self._season_len] - series[season_val_index]) / self._season_len
        return sum_val / self._season_len
        
    def _init_seasonal_coef(self, series):
        """ Инициализация начальных сезонных коэфициентов
        
        Параметры
        ---------
        series : pandas series
          Временной ряд
          
        Результат
        ---------
        seasonal_coef : numpy array
          Начальные сезонные коэфициенты
        """
        
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        # Здесь будут храниться начальные сезонные коэфициенты. Например, и количество равно 12, если сезонность 
        # годовая и данные представлены помесячно. 
        seasonal_coef = np.zeros(self._season_len)
        # кол-во сезонов
        n_seasons = int(len(series) / self._season_len)
        # Здесь храним средние значения для каждого сезона
        seasons_avg = []
        
        # Требуется посчитать среднее значение в рамках каждого сезона
        for season_num in range(n_seasons):
            season_avg_result = (series[self._season_len*season_num:self._season_len*(season_num+1)]).sum() / self._season_len
            seasons_avg.append(season_avg_result)

        # Нужно от каждого значения серии либо отнять среднее по сезону, либо поделить в зависимости от схемы ряда
        for season_val_index in range(self._season_len):
            season_coef_sum = 0.0
            for season_num in range(n_seasons):
                if self._model_type == 'add':
                    season_coef_sum += series[season_val_index+season_num*self._season_len] - seasons_avg[season_num]
                elif self._model_type == 'mul':
                    season_coef_sum += series[season_val_index+season_num*self._season_len] / seasons_avg[season_num]
              
            seasonal_coef[season_val_index] = season_coef_sum / n_seasons
            
        return seasonal_coef
    
    def fit_predict(self, series, n_preds):
        """ Инициализация начальных сезонных коэфициентов
        
        Параметры
        ---------
        series : pandas series
          Временной ряд
        n_preds : int
          Число желаемых предсказаний. Алгоритм корректно работает, если число кратно количеству длины одного сезона
          
        Результат
        ---------
        pкedict : list
          Сглаженные данные вместе с предсказанными
        """
        
        result = []
        
        seasonals = self._init_seasonal_coef(series)
        trend = self._init_trend_coef(series)
        smooth = series[0]
        result = [series[0]]
        
        for i in range(len(series)+n_preds):
            if i >= len(series): # Блок предсказания
                m = i - len(series) + 1
                if self._model_type == 'add':
                    result.append((smooth + m*trend) + seasonals[i%self._season_len])
                elif self._model_type == 'mul':
                    result.append((smooth + m*trend) * seasonals[i%self._season_len])
            else:
                val = series[i]
                
                if self._model_type == 'add':
                    last_smooth, smooth = smooth, self._alpha*(val-seasonals[i%self._season_len]) + (1-self._alpha)*(smooth+trend)
                elif self._model_type == 'mul':
                    last_smooth, smooth = smooth, self._alpha*(val/seasonals[i%self._season_len]) + (1-self._alpha)*(smooth+trend)
                
                trend = self._beta * (smooth-last_smooth) + (1-self._beta)*trend
                
                if self._model_type == 'add':
                    seasonals[i%self._season_len] = self._gamma*(val-smooth) + (1-self._gamma)*seasonals[i%self._season_len]
                elif self._model_type == 'mul':
                    seasonals[i%self._season_len] = self._gamma*(val/smooth) + (1-self._gamma)*seasonals[i%self._season_len]
                
                if self._model_type == 'add':
                    result.append(smooth+trend+seasonals[i%self._season_len])
                elif self._model_type == 'mul':
                    result.append((smooth+trend)*seasonals[i%self._season_len])
                
        return result