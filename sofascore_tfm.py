import pandas as pd
import requests, json
from json import JSONEncoder
from datetime import date
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
import numpy as np
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras import models, layers, losses, optimizers, metrics
import tensorflow as tf

from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse


BASE_PATH = 'http://sofascore-data-crawler.azurewebsites.net/data-miner'


def get_events(finished_events, req_date):

    if req_date:
      req = requests.get(BASE_PATH + '/events?finished=' + str(finished_events).lower() + '&date=' + req_date)
    else:
      req = requests.get(BASE_PATH + '/events?finished=' + str(finished_events).lower())

    # Generamos un dataframe en base a la respuesta obtenida por el API
    df_events = pd.DataFrame.from_dict(req.json())
    return df_events


def prepare_data(df_events, finished_events):

    # Eliminamos columnas innecesarias
    df_events = df_events.drop(columns=['id', 'status', 'groundType'], axis=1)

    # Generamos el dataframe con los datos del jugador en casa
    drop_player_columns = ['id', 'sport', 'name', 'shortName', 'fullName',
                           'nameCode', 'residence', 'birthplace', 'country', 'gender']

    df_homeplayer = pd.DataFrame.from_records(df_events['homePlayer'])
    df_homeplayer = df_homeplayer.drop(columns=drop_player_columns, axis=1)
    df_homeplayer.columns = [
        str(col) + '_homePlayer' for col in df_homeplayer.columns]

    # Generamos el dataframe con los datos del jugador en casa
    df_awayplayer = pd.DataFrame.from_records(df_events['awayPlayer'])
    df_awayplayer = df_awayplayer.drop(columns=drop_player_columns, axis=1)
    df_awayplayer.columns = [
        str(col) + '_awayPlayer' for col in df_awayplayer.columns]

    # Generamos el dataframe con la puntuación obtenida por set por el jugador en casa
    if(finished_events):
        df_homescore = pd.DataFrame.from_records(df_events['homeScore'])
        df_homescore.columns = [
            str(col) + '_home' for col in df_homescore.columns]

    # Generamos el dataframe con la puntuación obtenida por set por el jugador fuera
    if(finished_events):
        df_awayscore = pd.DataFrame.from_records(df_events['awayScore'])
        df_awayscore.columns = [
            str(col) + '_away' for col in df_awayscore.columns]

    df_events = df_events.drop(
        columns=['homePlayer', 'awayPlayer', 'homeScore', 'awayScore'], axis=1)

    if (finished_events):
        df_events = pd.concat(
            [df_events, df_homeplayer, df_awayplayer, df_homescore, df_awayscore], axis=1)
    else:
        df_events = pd.concat(
            [df_events, df_homeplayer, df_awayplayer], axis=1)

    # Mapping table:
    # right-handed --> 1
    # left-handed --> 0
    # ground_type --> Clay:1, Grass:0
    # winner --> home:1, away:0

    # Set way of play as a numeric variable
    df_events['plays_homePlayer'] = df_events['plays_homePlayer'].replace(
        ['right-handed', 'left-handed'], [1, 0])
    df_events['plays_awayPlayer'] = df_events['plays_awayPlayer'].replace(
        ['right-handed', 'left-handed'], [1, 0])

    # Set groundType as a numeric variable
    #df_events['groundType'] = df_events['groundType'].replace(['Clay', 'Grass'],[1, 0])

    # Set winner as a numeric variable
    if(finished_events):
        for index, row in df_events.iterrows():
            df_events.at[index,
                         'winner'] = 1 if row['winner'] == row['slug_homePlayer'] else 0
            if row['current_home'] == 3 or row['current_away'] == 3:
                df_events.drop(index, inplace=True)

    # Finalmente, eliminamos el resto de columnas que no vamos a necesitar para el entrenamiento"""

    df_events = df_events.drop(columns=['slug', 'slug_homePlayer', 'slug_awayPlayer', 'date',
                               'firstToServe', 'doubles', 'national_homePlayer', 'national_awayPlayer'], axis=1)

    """
  Llegados a este punto, todo el dataframe contiene únicamente variables numéricas. Tenemos que revisar ahora si contiene valores nulos o no informados y qué hacer.

  A priori, se realizará la siguiente operación sobre las columnas que contengan valores nulos:

  * weight --> Se sustituirá por la media
  * height --> Se sustituirá por la media
  * turnedPro --> Se sustituirá por la media
  * prizeTotal --> Se sustituirá por la media
  * ranking --> Se sustituirá por la media

  """

    # plays
    mode_plays_home = pd.to_numeric(
        df_events['plays_homePlayer'], errors='coerce').mode()
    mode_plays_away = pd.to_numeric(
        df_events['plays_awayPlayer'], errors='coerce').mode()
    df_events['plays_homePlayer'] = df_events['plays_homePlayer'].fillna(1)
    df_events['plays_awayPlayer'] = df_events['plays_awayPlayer'].fillna(1)

    # TurnedPro
    meanTurnedPro_home = pd.to_numeric(
        df_events['turnedPro_homePlayer'], errors='coerce').mean()
    meanTurnedPro_away = pd.to_numeric(
        df_events['turnedPro_awayPlayer'], errors='coerce').mean()
    df_events['turnedPro_homePlayer'] = df_events['turnedPro_homePlayer'].fillna(
        meanTurnedPro_home)
    df_events['turnedPro_awayPlayer'] = df_events['turnedPro_awayPlayer'].fillna(
        meanTurnedPro_away)

    # prizeTotal
    meanPrizeTotal_home = pd.to_numeric(
        df_events['prizeTotal_homePlayer'], errors='coerce').mean()
    meanPrizeTotal_away = pd.to_numeric(
        df_events['prizeTotal_awayPlayer'], errors='coerce').mean()
    df_events['prizeTotal_homePlayer'] = df_events['prizeTotal_homePlayer'].fillna(
        meanPrizeTotal_home)
    df_events['prizeTotal_awayPlayer'] = df_events['prizeTotal_awayPlayer'].fillna(
        meanPrizeTotal_away)

    # prizeCurrent
    meanPrizeCurrent_home = pd.to_numeric(
        df_events['prizeCurrent_homePlayer'], errors='coerce').mean()
    meanPrizeCurrent_away = pd.to_numeric(
        df_events['prizeCurrent_awayPlayer'], errors='coerce').mean()
    df_events['prizeCurrent_homePlayer'] = df_events['prizeCurrent_homePlayer'].fillna(
        meanPrizeCurrent_home)
    df_events['prizeCurrent_awayPlayer'] = df_events['prizeCurrent_awayPlayer'].fillna(
        meanPrizeCurrent_away)

    # prizeCurrent
    meanPoints_home = pd.to_numeric(
        df_events['points_homePlayer'], errors='coerce').mean()
    meanPoints_away = pd.to_numeric(
        df_events['points_awayPlayer'], errors='coerce').mean()
    df_events['points_homePlayer'] = df_events['points_homePlayer'].fillna(
        meanPoints_home)
    df_events['points_awayPlayer'] = df_events['points_awayPlayer'].fillna(
        meanPoints_away)

    # weigth
    weight_home = pd.to_numeric(
        df_events['weight_homePlayer'], errors='coerce').mean()
    weight_away = pd.to_numeric(
        df_events['weight_awayPlayer'], errors='coerce').mean()
    df_events['weight_homePlayer'] = df_events['weight_homePlayer'].fillna(
        weight_home)
    df_events['weight_awayPlayer'] = df_events['weight_awayPlayer'].fillna(
        weight_away)

    # heigth
    height_home = pd.to_numeric(
        df_events['height_homePlayer'], errors='coerce').mean()
    height_away = pd.to_numeric(
        df_events['height_awayPlayer'], errors='coerce').mean()
    df_events['height_homePlayer'] = df_events['height_homePlayer'].fillna(
        height_home)
    df_events['height_awayPlayer'] = df_events['height_awayPlayer'].fillna(
        height_away)

    # ranking
    ranking_home = pd.to_numeric(
        df_events['ranking_homePlayer'], errors='coerce').mean()
    ranking_away = pd.to_numeric(
        df_events['ranking_awayPlayer'], errors='coerce').mean()
    df_events['ranking_homePlayer'] = df_events['ranking_homePlayer'].fillna(
        ranking_home)
    df_events['ranking_awayPlayer'] = df_events['ranking_awayPlayer'].fillna(
        ranking_away)

    # bestRanking
    best_ranking_home = pd.to_numeric(
        df_events['bestRanking_homePlayer'], errors='coerce').mean()
    best_ranking_away = pd.to_numeric(
        df_events['bestRanking_awayPlayer'], errors='coerce').mean()
    df_events['bestRanking_homePlayer'] = df_events['bestRanking_homePlayer'].fillna(
        best_ranking_home)
    df_events['bestRanking_awayPlayer'] = df_events['bestRanking_awayPlayer'].fillna(
        best_ranking_away)

    # previousRanking
    previous_ranking_home = pd.to_numeric(
        df_events['previousRanking_homePlayer'], errors='coerce').mean()
    previous_ranking_away = pd.to_numeric(
        df_events['previousRanking_awayPlayer'], errors='coerce').mean()
    df_events['previousRanking_homePlayer'] = df_events['previousRanking_homePlayer'].fillna(
        best_ranking_home)
    df_events['previousRanking_awayPlayer'] = df_events['previousRanking_awayPlayer'].fillna(
        best_ranking_away)

    # tournamentsPlayed
    mean_tournamentsPlayed_home = pd.to_numeric(
        df_events['tournamentsPlayed_homePlayer'], errors='coerce').mean()
    mean_tournamentsPlayed_away = pd.to_numeric(
        df_events['tournamentsPlayed_awayPlayer'], errors='coerce').mean()
    df_events['tournamentsPlayed_homePlayer'] = df_events['tournamentsPlayed_homePlayer'].fillna(
        mean_tournamentsPlayed_home)
    df_events['tournamentsPlayed_awayPlayer'] = df_events['tournamentsPlayed_awayPlayer'].fillna(
        mean_tournamentsPlayed_away)

    # birthDateTimestamp
    mean_birthDateTimestamp_home = pd.to_numeric(
        df_events['birthDateTimestamp_homePlayer'], errors='coerce').mean()
    mean_birthDateTimestamp_away = pd.to_numeric(
        df_events['birthDateTimestamp_awayPlayer'], errors='coerce').mean()
    df_events['birthDateTimestamp_homePlayer'] = df_events['birthDateTimestamp_homePlayer'].fillna(
        mean_birthDateTimestamp_home)
    df_events['birthDateTimestamp_awayPlayer'] = df_events['birthDateTimestamp_awayPlayer'].fillna(
        mean_birthDateTimestamp_away)

    # sets information
    if(finished_events):
        df_events['period1_home'] = df_events['period1_home'].fillna(0)
        df_events['period2_home'] = df_events['period2_home'].fillna(0)
        df_events['period3_home'] = df_events['period3_home'].fillna(0)
        df_events['period4_home'] = df_events['period4_home'].fillna(0)
        df_events['period5_home'] = df_events['period5_home'].fillna(0)
        df_events['period6_home'] = df_events['period6_home'].fillna(0)
        df_events['period1TieBreak_home'] = df_events['period1TieBreak_home'].fillna(
            0)
        df_events['period2TieBreak_home'] = df_events['period2TieBreak_home'].fillna(
            0)
        df_events['period3TieBreak_home'] = df_events['period3TieBreak_home'].fillna(
            0)
        df_events['period4TieBreak_home'] = df_events['period4TieBreak_home'].fillna(
            0)
        df_events['period5TieBreak_home'] = df_events['period5TieBreak_home'].fillna(
            0)
        df_events['period6TieBreak_home'] = df_events['period6TieBreak_home'].fillna(
            0)

        df_events['period1_away'] = df_events['period1_away'].fillna(0)
        df_events['period2_away'] = df_events['period2_away'].fillna(0)
        df_events['period3_away'] = df_events['period3_away'].fillna(0)
        df_events['period4_away'] = df_events['period4_away'].fillna(0)
        df_events['period5_away'] = df_events['period5_away'].fillna(0)
        df_events['period6_away'] = df_events['period6_away'].fillna(0)
        df_events['period1TieBreak_away'] = df_events['period1TieBreak_away'].fillna(
            0)
        df_events['period2TieBreak_away'] = df_events['period2TieBreak_away'].fillna(
            0)
        df_events['period3TieBreak_away'] = df_events['period3TieBreak_away'].fillna(
            0)
        df_events['period4TieBreak_away'] = df_events['period4TieBreak_away'].fillna(
            0)
        df_events['period5TieBreak_away'] = df_events['period5TieBreak_away'].fillna(
            0)
        df_events['period6TieBreak_away'] = df_events['period6TieBreak_away'].fillna(
            0)

    """Añadimos una nueva columna cuyo valor reflejará el resultado del partido. La calcularemos a partir del número de sets ganados por el jugador local y el visitante, de modo que tendremos 4 posibles valores: 2-0, 2-1, 1-2, 0-2.

  Esta variable será la etiqueta que tendrá que predecir la red neuronal
  """

    # Estos son los 4 valores posibles:
    # 1 --> 2-0 gana local
    # 0.66 --> 2-1 gana local
    # 0.33 --> 1-2 gana visitante
    # 0 --> 2-0 gana visitante

    # Para ello, iteramos sobre el dataframe y vamos calculando los valores
    if(finished_events):
        for index, row in df_events.iterrows():
            score = None
            if row['current_home'] == 2 and row['current_away'] == 0:
                score = 1
            elif row['current_home'] == 2 and row['current_away'] == 1:
                #score = 0.66
                score = 1
            elif row['current_home'] == 1 and row['current_away'] == 2:
                #score = 0.33
                score = 0
            elif row['current_home'] == 0 and row['current_away'] == 2:
                score = 0
            df_events.at[index, 'winner_score'] = score

    if (finished_events):
        df_events = df_events.drop(columns=['winner',
                                            'period1_home',
                                            'period2_home',
                                            'period3_home',
                                            'period4_home',
                                            'period5_home',
                                            'period6_home',
                                            'period1_away',
                                            'period2_away',
                                            'period3_away',
                                            'period4_away',
                                            'period5_away',
                                            'period6_away',
                                            'period1TieBreak_home',
                                            'period2TieBreak_home',
                                            'period3TieBreak_home',
                                            'period4TieBreak_home',
                                            'period5TieBreak_home',
                                            'period6TieBreak_home',
                                            'period1TieBreak_away',
                                            'period2TieBreak_away',
                                            'period3TieBreak_away',
                                            'period4TieBreak_away',
                                            'period5TieBreak_away',
                                            'period6TieBreak_away',
                                            'current_home',
                                            'current_away'], axis=1)
    else:
        df_events = df_events.drop(columns=['winner'], axis=1)

    return df_events


def build_model(df_events):

    X = df_events.iloc[:, :-1]
    y = df_events.iloc[:, -1:]

    size_input_data = len(X.columns)

    y = np.reshape(y, (-1, 1))
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    print(scaler_x.fit(X))
    x_scale = scaler_x.transform(X)
    print(scaler_y.fit(y))
    y_scale = scaler_y.transform(y)

    # Separamos en conjunto de entrenamiento y conjunto de test
    X_train, X_test, y_train, y_test = train_test_split(x_scale, y_scale)

    model = Sequential()

    model.add(Dense(size_input_data, input_dim=size_input_data))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # model.add(Dense(1000))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.3))

    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('linear'))

    return model, X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):

    # Configuramos el learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    loss = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE)

    # Compilamos el modelo
    model.compile(loss=loss, optimizer=optimizer, metrics=['mse', 'mae'])

    # Entrenamos la red neuronal
    history = model.fit(X_train, y_train, epochs=120,
                        batch_size=40, verbose=1, validation_split=0.2)

    return model


def evaluate_model(model, X_train, y_train):

    scores = model.evaluate(X_train, y_train)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # ------------------------------------------------
    # Pérdida de entrenamiento y validación por epoch con zoom
    # ------------------------------------------------
    # -----------------------------------------------------------
    # Recuperar una lista de resultados de la lista de datos de entrenamiento y pruebas para cada epoch de entrenamiento
    # -----------------------------------------------------------
    loss = history.history['loss']
    mae = history.history['mae']
    epochs = range(len(loss))  # Get number of epochs

    zoomed_loss = loss[0:30]
    zoomed_mae = mae[0:30]
    zoomed_epochs = range(0, 30)
    #
    # tu código para el plot con zoom del ejercicio 8 aquí
    plt.figure(figsize=(10, 8))
    plt.plot(zoomed_epochs, zoomed_mae, 'r')
    plt.plot(zoomed_epochs, zoomed_loss, 'b')
    plt.title('MAE and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["MAE", "Loss"])

    plt.figure()


def plot_prediction(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


class Predictions(Resource):

    def get(self):

        print('Getting future events...')
        df = get_events(False, '2021-07-15')
        matches = df[['slug', 'date', 'homePlayer', 'awayPlayer']]
        df = prepare_data(df, False)
        
        # load the saved model
        json_file = open('model.json', 'r')
        loaded_nn_model_json = json_file.read()
        json_file.close()
        loaded_nn_model = model_from_json(loaded_nn_model_json)
        # load weight into classification model
        loaded_nn_model.load_weights('model_weights.h5')

        X = df.iloc[:]
        scaler_x = MinMaxScaler()
        scaler_x.fit(X)
        x_scale = scaler_x.transform(X)

        print('Prediction triggered!!!!')
        predictions = loaded_nn_model.predict(x_scale)
        
        predictions_result = {}

        for index, row in matches.iterrows():
          match = Match(row['date'], row['homePlayer']['name'], row['awayPlayer']['name'], predictions[index])
          if row['date'] in predictions_result:
            predictions_result[row['date']].append(match.to_json())
          else:
            predictions_result[row['date']] = [match.to_json()]
            
        return jsonify(**predictions_result)
    pass
  

class Match():
  
  def __init__(self, date, home_player, away_player, prediction):
    self.date = date
    self.home_player = home_player
    self.away_player = away_player
    self.prediction_prob = str(prediction).replace('[', '').replace(']', '')
    self.predicted_winner = self.calculate_winner(prediction)
    
  def calculate_winner(self, prediction):
    return self.home_player if prediction > 0.5 else self.away_player
  
  def to_json(self):
    return {
      "homePlayer": self.home_player,
      "awayPlayer": self.away_player,
      "date": self.date,
      "predictionProbability": self.prediction_prob,
      "predictedWinner": self.predicted_winner
    }

  

app = Flask(__name__)
api = Api(app)

api.add_resource(Predictions, '/predictions')


if __name__ == '__main__':

    print('Get list with past matches...')
    df = get_events(True, None)
    df = prepare_data(df, True)
    
    print('Training neural network')
    model, X_train, X_test, y_train, y_test = build_model(df)
    model = train_model(model, X_train, y_train)

    # Serialize model to JSON
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights('model_weights.h5')
    print('Saved model to disk')

    app.run()  # run our Flask app
