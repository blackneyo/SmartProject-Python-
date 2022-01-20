# Import Data Preprocessing & Graph Library
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import plotly.graph_objects as go

# Import Predictive Analytics Model Library
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model, model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

from fbprophet import Prophet


class PredictModel:
    def __init__(self):
        pass
    
    # 회사명 -> 종목코드명, Finace-datareader로 주가데이터 받기
    def start_predict(self, theme, company, modelString, predict_term):
        code = ''
        from_date = '2020-01-20'

        # 여기서 테마, 회사 를 조건으로 종목코드 넣기
        if theme == 'game':
            if company == '엔씨소프트':
                code = '036570'
            if company == '넷마블':
                code = '251270'
            if company == '펄어비스':
                code = '263750'
        if theme == 'broadCast':
            if company == 'CJ ENM':
                code = '035760'
            if company == '스튜디오드래곤':
                code = '253450'
            if company == 'SBS':
                code = '034120'
        if theme == 'realEstate':
            if company == '롯데리츠':
                code = '330590'
            if company == 'SK디앤디':
                code = '210980'
            if company == '신한알파리츠':
                code = '293940'
        if theme == 'sea':
            if company == 'HMM':
                code = '011200'
            if company == 'KSS해운':
                code = '044450'
            if company == '와이엔텍':
                code = '067900'
        if theme == 'pharmaceutical':
            if company == '셀트리온':
                code = '068270'
            if company == '한미약품':
                code = '128940'
            if company == '삼성바이오로직스':
                code = '207940'

        print(theme, company, modelString)
        df = fdr.DataReader(code, from_date)

        if modelString == 'lstm':
            accur_print, cost_print = self.lstm(df, predict_term, company)
            return accur_print, cost_print    

        if modelString == 'prophet':
            accur_print, cost_print = self.prophet(df, predict_term)
            return accur_print, cost_print

    ### LSTM 분석모델 ###
    def lstm(self, df, predict_term, company):
        
        # Data preprocessing
        stock = df.copy()
        stock['Volume'] = stock['Volume'].replace(0, np.nan)
        stock = stock.dropna()

        # Normalizing
        scaler = MinMaxScaler()
        scaled_stock = scaler.fit_transform(stock['Close'].values.reshape(-1,1))

        # Train data, Test data
        close_data = scaled_stock

        split_percent = 0.70
        split = int(split_percent * len(close_data))

        close_train = close_data[:split]
        close_test = close_data[split:]

        date_train = stock.index[:split]
        date_test = stock.index[split:]

        look_back = 5

        train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=32)
        test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

        # LSTM Modeling
        model = Sequential()
        model.add(
            LSTM(50,
                 activation='elu',
                 return_sequences=True,
                 input_shape=(look_back, 1))
        )
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        sgd = tf.keras.optimizers.SGD(learning_rate=0.01,
                                decay=1e-7,
                                momentum=0.9,
                                nesterov=False)
        model.compile(optimizer=sgd, loss='mean_squared_error')

        ############################# save ###################################
        # h5 = 'data/lstm_model/LSTM_{}.h5'.format(company)
        # checkpoint = ModelCheckpoint(h5,
        #                         monitor='loss',
        #                         verbose=1,
        #                         save_best_only=True,
        #                         save_weights_only=True,
        #                         mode='auto')
        # early_stop = EarlyStopping(monitor='loss',
        #                         patience=20)
        # json = 'data/lstm_model/LSTM_{}.json'.format(company)
        # model_json = model.to_json()
        # with open(json, 'w') as json_file:
        #     json_file.write(model_json)

        # num_epochs = 100
        # model.fit(train_generator, validation_data=test_generator, epochs=num_epochs, verbose=1,
        #                 callbacks=[checkpoint, early_stop])
        ############################## load ####################################
        def load_lstm_model(json, h5):
            json_file = open(json, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(h5)
            return model
        model = load_lstm_model('data/lstm_model/LSTM_{}.json'.format(company),
                            'data/lstm_model/LSTM_{}.h5'.format(company))
        ##########################################################################
        # Prediction
        prediction = model.predict(test_generator)

        close_data = close_data.reshape((-1))
        close_train = close_train.reshape((-1))
        close_test = close_test.reshape((-1))
        prediction = prediction.reshape((-1))

        # 평균절대값백분율오차계산 (MAPE)
        mape = mean_absolute_percentage_error(prediction, close_test[look_back:])
        accuracy = (1 - mape) * 100
        accur_print = '정확도: ' + str(round(accuracy, 2)) + '%'

        # Forecasting
        def predict(num_prediction, model):
            prediction_list = close_data[-look_back:]

            for _ in range(num_prediction):
                x = prediction_list[-look_back:]
                x = x.reshape((1, look_back, 1))
                out = model.predict(x)[0][0]
                prediction_list = np.append(prediction_list, out)
            prediction_list = prediction_list[look_back - 1:]

            return prediction_list

        def predict_dates(num_prediction):
            last_date = df.index.values[-1]
            prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
            return prediction_dates

        num_prediction = 20

        if predict_term == '5':
            num_prediction = 5
        if predict_term == '30':
            num_prediction = 30

        forecast = predict(num_prediction, model)
        forecast_dates = predict_dates(num_prediction)

        # Chart drawing
        inverse_close_train = scaler.inverse_transform(close_train.reshape(-1,1))
        inverse_close_data = scaler.inverse_transform(close_data.reshape(-1,1))
        inverse_prediction = scaler.inverse_transform(prediction.reshape(-1,1))
        inverse_forecast = scaler.inverse_transform(forecast.reshape(-1,1))
        chart_close_train = inverse_close_train.reshape((-1))
        chart_close_data = inverse_close_data.reshape((-1))
        chart_prediction = inverse_prediction.reshape((-1))
        chart_forecast = inverse_forecast.reshape((-1))

        trace1 = go.Scatter(
            x=date_train,
            y=chart_close_train,
            mode='lines',
            name='TrainData'
        )
        trace2 = go.Scatter(
            x=stock.index[split-1:],
            y=chart_close_data[split-1:],
            mode='lines',
            name='TestData'
        )
        trace3 = go.Scatter(
            x=date_test[look_back:],
            y=chart_prediction,
            mode='lines',
            name='Prediction'
        )
        trace4 = go.Scatter(
            x=forecast_dates,
            y=chart_forecast,
            mode='lines',
            name='Forecast'
        )
        layout = go.Layout(
            margin=dict(l=20, r=20, t=10, b=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis={'title': "날짜"},
            yaxis={'title': "종가"}
        )
        # Save chart to json
        fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
        fig.write_json('chart.json')

        # Print tomorrow cost
        tomorrow_cost = scaler.inverse_transform(forecast.reshape(-1,1))
        cost_print = '(익일)예측가: ' + '{:,}'.format(int(tomorrow_cost[1,0])) + '원' 
        return accur_print, cost_print

    ### Prophet 분석모델 ###
    def prophet(self, df, predict_term):

        # Data preprocessing
        stock = df.copy()
        stock['Volume'] = stock['Volume'].replace(0, np.nan)
        stock = stock.dropna()

        stock['y'] = stock['Close']  
        stock['ds'] = stock.index
    
        # Prophet Modeling
        m = Prophet()  
        m.fit(stock)  

        num_prediction = 20

        if predict_term == '5':
            num_prediction = 5
        if predict_term == '30':
            num_prediction = 30

        future = m.make_future_dataframe(periods=num_prediction, freq='B')       
        forecast = m.predict(future)

        # Value correction
        def error_improv(x):
            error = abs(stock['y'][stock.shape[0]-1] - forecast['yhat'][stock.shape[0]-1])
            if stock['y'][stock.shape[0]-1] > forecast['yhat'][stock.shape[0]-1]:
                x = x + error
            else:
                x = x - error
            return x

        # Chart drawing
        forecast['yhat'][stock.shape[0]-1:] = forecast['yhat'][stock.shape[0]-1:].apply(error_improv)

        trace1 = go.Scatter(
            x = stock['ds'],
            y = stock['y'],
            mode = 'lines',
            name = 'TrainData'
        )
        trace2 = go.Scatter(
            x = forecast['ds'][stock.shape[0]-1:],
            y = forecast['yhat'][stock.shape[0]-1:],
            mode = 'lines',
            name = 'Forecast'
        )
        trace3 = go.Scatter(
            x = forecast['ds'][:stock.shape[0]],
            y = forecast['yhat'][:stock.shape[0]],
            mode='lines',
            name = 'Prediction'
        )
        layout = go.Layout(
            margin=dict(l=20, r=20, t=10, b=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis={'title': "날짜"},
            yaxis={'title': "종가"}
        )
        # Save chart to json
        fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
        fig.write_json('chart.json')

        # MAPE
        mape = mean_absolute_percentage_error(stock['y'], forecast['yhat'][:stock.shape[0]])
        accuracy = (1 - mape) * 100
        accur_print = '정확도: ' + str(round(accuracy, 2)) + '%'

        # Print tomorrow cost
        tomorrow_cost = forecast['yhat'][stock.shape[0]]
        cost_print = '(익일)예측가: ' + '{:,}'.format(int(tomorrow_cost)) + '원' 
        return accur_print, cost_print