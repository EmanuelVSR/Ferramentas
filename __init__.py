#Funções:
# PresentValue(list, rate): calcula o valor presente de uma lista de fluxos 'list'
# a partir de uma taxa de desconto 'rate'

#MA(nPeriods, dfname, origin): calcula uma média móvel simples
#   'nPeriods': número de períodos da média móvel. ex: 100
#   'dfname': nome do dataframe onde está a série de preços base e estará a média móvel. ex: df
#   'origin': nome da coluna onde está a série de preços base. ex: 'MGLU3.SA'

#GetDFPrices(AcoesBR, IndicesOuEUA, start, end, Ptype): retorna um dataframe com n séries de preços.
#   'AcoesBR': lista de tickers de ações brasileiras (que o Yahoo Finance precisa de '.SA' para interpretar), ex: ['JBSS3'] ou vazio []
#   'IndicesOuEUA': lista de tickers de ações ou ickers que não recebem '.SA'. ex: ['^BVSP','AAL']
#   'start': string da data de início da série. ex: 'DD/MM/AAAA' ou '5/12/2013'
#   'end': string da data final da série. ex: 'DD/MM/AAAA' ou '5/12/2013' ou '' para dia de hoje.
#   'Ptype': preço a ser retornado. 'Adj Close' OU 'Close'

#PVol(dfname, mode=): retorna a volatilidade anual de uma série de retornos
#   'dfname': nome do DataFrame
#   'mode': opcional, deixar vazio caso seja uma série de retornos. 'Price' caso seja uma série de preços

#PVar(dfname, mode=): retorna a variância de uma série de retornos.
#   'dfname': nome do dataframe
#   'mode': opcional, deixar vazio caso seja uma série de retornos. 'Price' caso seja uma série de preços

#PLogR(dfname): retorna os log retornos de um dataframe de preços
#   'dfname': nome do dataframe

#GetBacenData(Titulos, codigos_bcb): retorna um df com os dados retirados do API do BCB
#   'Titulos': lista de dados
#   'codigos_bcb': lista de códigos na mesma ordem que os Títulos
#
#   Códigos podem ser encontrados no site https://www3.bcb.gov.br/sgspub/localizarseries/localizarSeries.do?method=prepararTelaLocalizarSeries

#Objetos:
#series(ticker, Ptype, start, end): retorna a série de preços de uma ação
#   'ticker': ticker da ação ou índice no formato Yahoo. ex: 'JBSS3.SA', '^BVSP'
#   'Ptype': preço a ser retornado. 'Adj Close' OU 'Close'
#   'start': string da data de início da série. ex: 'DD/MM/AAAA' ou '5/12/2013'
#   'end': string da data final da série. ex: 'DD/MM/AAAA' ou '5/12/2013' ou '' para dia de hoje.
#   "self.prices" = série de preços
#   "self.ticker" = 'ticker'
#   "self.Ptype" = 'Ptype'
#   DropColumns(self): elimina todas as colunas que não são 'Data' ou Ptype. Muda "self.prices"
#   DropNaN(self): elimina valores em branco ou inválidos. Muda "self.prices"
#   Round(self): arredonda a coluna de preços. Muda "self.prices"
#   Treat(self): executa as 3 funções acima, muda o nome da série para "self.ticker" e retorna "self.prices"
#   ToCSV(self): salva a série de preços para um arquivo CSV chamado "self.ticker".csv

import pandas as pd
import pandas_datareader.data as web
import datetime as dt
from datetime import date
import numpy as np
import math

###########################################################################################################
def PresentValue(list, rate):
    a = 0
    x = 0
    while a < len(list):
        x = x + list[a]/pow(1 + rate, a)
        a += 1
    x = round(x, 2)
    return x

##########################################################################################################
##########################################################################################################
class series:
    def __init__(self, ticker, Ptype, start, end):
        self.ticker = ticker
        self.Ptype = Ptype
        self.start = start
        self.end = end

        if self.Ptype == 'Close':
            self.xyz = ['High', 'Low', 'Open', 'Adj Close', 'Volume']
        else:
            self.xyz = ['High', 'Low', 'Open', 'Close', 'Volume']

        self.prices = web.DataReader(ticker, 'yahoo', self.start, self.end)

    def DropColumns(self):
        self.prices.drop(self.xyz, axis=1, inplace=True)

    def DropNaN(self):
        self.prices.dropna(subset=[self.Ptype], axis=0, inplace=True)

    def Round(self):
        self.prices[self.Ptype] = self.prices[self.Ptype].round(2)

    def ToCSV(self):
        self.prices.to_csv(self.ticker + '.csv')

    def Treat(self):
        self.DropNaN()
        self.DropColumns()
        self.Round()
        self.prices.columns = [self.ticker]
        return self.prices

#################################################################################
def GetDFPrices(AcoesBR, IndicesOuEUA, start, end, Ptype):
    tickers = []

    sday, smonth, syear = map(int, start.split('/'))
    start = dt.datetime(syear, smonth, sday)

    if end == '':
        end = date.today()

    else:
        eday, emonth, eyear = map(int, end.split('/'))
        end = dt.datetime(eyear, emonth, eday)

    AcoesBR.sort()
    IndicesOuEUA.sort()

    if len(IndicesOuEUA) != 0:
        for item in IndicesOuEUA:
            tickers.append(item)

    if len(AcoesBR) != 0:
        for item in AcoesBR:
            tickers.append(item + '.SA')

    df1 = series(tickers[0], Ptype, start, end)
    df1 = df1.Treat()

    i = 1

    while i < len(tickers):
        s = series(tickers[i], Ptype, start, end).Treat()
        df1 = df1.join(s)

        i += 1

    return df1

#####################################################################################################
#####################################################################################################
def MA(nPeriods, dfname, origin):
    dfname[str(nPeriods) + 'ma'] = dfname[str(origin)].rolling(window=nPeriods, min_periods=0).mean()
    dfname[str(nPeriods) + 'ma'] = dfname[str(nPeriods) + 'ma'].round(2)

#####################################################################################################
#####################################################################################################
def PVol(dfname, mode=None):
    if mode is None:
        return dfname.std()*np.sqrt(252)
    if mode == 'Price':
        return dfname.pct_change().std()*np.sqrt(252)

#####################################################################################################
#####################################################################################################
def PVar(dfname, mode=None):
    if mode is None:
        return dfname.var()
    if mode == 'Price':
        return dfname.pct_change().var()

#####################################################################################################
#####################################################################################################
def PLogR(dfname):
    return dfname.pct_change().apply(lambda x: np.log(1+x))

#####################################################################################################
#####################################################################################################
def GetFredData(ticker, start, end):
    sday, smonth, syear = map(int, start.split('/'))
    start = dt.datetime(syear, smonth, sday)

    if end == '':
        end = date.today()

    else:
        eday, emonth, eyear = map(int, end.split('/'))
        end = dt.datetime(eyear, emonth, eday)

    return web.DataReader(ticker, 'fred', start, end)

#####################################################################################################
#####################################################################################################
def GetBacenData(Titulos, codigos_bcb):
    main_df = pd.DataFrame()

    for i, codigo in enumerate(codigos_bcb):
        url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(str(codigo))
        df = pd.read_json(url)

        df['DATE'] = pd.to_datetime(df['data'], dayfirst=True)
        df.drop('data', axis=1, inplace=True)
        df.set_index('DATE', inplace=True)
        df.columns = [str(Titulos[i])]

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.merge(df, how='outer', left_index=True, right_index=True)

    main_df.fillna(method='ffill', inplace=True)

    return main_df
