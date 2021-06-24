
import math
from numpy.lib.function_base import _i0_1
import pandas_datareader as web
import numpy as np
import pandas as pd
import config as cfg
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import datetime
inputSize = 60

df = web.DataReader('BITFINEX/BTCUSD', data_source='quandl',start= '2012-01-01',api_key= '%%YOUR API KEY HERE%%')
#Só nos interessa o preço de fechamento
data = df.filter(['Last'])
data = data.sort_values(by=['Date'])
#Queremos na forma numpy.
dataset = data.values
# 60% dos dados são utilizados para fazer o conjunto de treinamento
trainingDataLen = math.ceil( len(dataset)*0.6 )
scaler = MinMaxScaler(feature_range= (0,1)) #Escalar todos os valores entre 0 and 1
scaledData = scaler.fit_transform(dataset)
#Criar o conjunto de treino
trainData = scaledData[0:trainingDataLen, :]
#Dividí-lo em X = série de pontos em que cada ponto é um array com 60 preços anteriores; Y = série de pontos em que cada ponto é o preço atual
xTrain = [] #indp
yTrain = [] #depd
for i in range(inputSize, len(trainData)):
    xTrain.append(trainData[i-inputSize:i,0])
    yTrain.append(trainData[i,0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)

# O LSTM requer que se faça reshape dos dados como feito a seguir numofSamples, numofTimesteps, numofFeatures
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1],1))

#Construir o LSTM:
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (xTrain.shape[1],1)))
#50 neurons, return_sequences=True signifca que há uma próxima camada LSTM, e o formato da entrada é de 60 preços e 1 saída (preço atual)
model.add(LSTM(50,return_sequences=False))
#50 neurons novamente, agora falso porque não há uma próxima LSTM
model.add(Dense(25)) #25 neurons, camada regular
model.add(Dense(1)) #1 neuron de saída com o preço
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xTrain, yTrain, batch_size=1, epochs=3)


#Array com conjunto de teste (os 40% restante dos dados)
testData = scaledData[trainingDataLen-inputSize: , :]
testDataIncludeFut = testData
xTest = []
xFutures = []
yTest = dataset[trainingDataLen: , :]
for i in range(inputSize, len(testData)):
    xTest.append(testData[i-inputSize:i, 0])

xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0],xTest.shape[1],1))

#Get the models predicted price values:
validvals = model.predict(xTest)
validvals = scaler.inverse_transform(validvals)
#Erro quadrático médio
rmse=np.sqrt(np.mean(((validvals- yTest)**2)))
print('rmse=',rmse)

#Valores futuros:
FUTURE_SPAN  = 30 #número de dias que o código vai tentar prever o preço:
import copy
testDataCopy = copy.deepcopy(testData)
testDataIncludeFut = copy.deepcopy(testData)
xFutures = []
for i in range(FUTURE_SPAN):
    xFutures.append(testDataIncludeFut[-1-inputSize:-1, 0])
    xFuturesTransf = copy.deepcopy(np.array(xFutures))
    xFuturesTransf = np.reshape(xFuturesTransf, (xFuturesTransf.shape[0],xFuturesTransf.shape[1],1))
    yFutures = model.predict(xFuturesTransf)
    testDataIncludeFut = np.append(testDataCopy, yFutures, axis=0)
predvals = scaler.inverse_transform(yFutures)

#Plotar os dados:
train = data[:trainingDataLen]
valid = data[trainingDataLen:]
valid['Validvals'] = validvals
pred = pd.DataFrame()
# pred.append(pd.DataFrame({'Date': pd.date_range(start=valid.Date.iloc[-1], periods=6, freq='D', closed='right')}, {'Validvals':[predvals[:,0]]}))
pred['Validvals'] = predvals[:,0]
pred['Date'] = pd.Series([(datetime.datetime.today()+datetime.timedelta(days=(i+1))).strftime('%Y-%m-%d') for i in range(FUTURE_SPAN)], index=pred.index)
pred.set_index('Date', inplace=True)
validprev = valid.append(pred)
#Visualize
plt.figure(figsize=(8,4))
plt.title('Model')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price USD($)', fontsize=12)
plt.plot(train['Last'])
plt.plot(validprev[['Last','Validvals']])
plt.legend(['Conjunto de treino','Conjunto de validação','Valor Calculado'], loc='upper left')
plt.show()