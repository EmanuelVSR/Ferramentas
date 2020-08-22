#df.pct_change() muda todas as colunas para retornos percentuais
import Impactus_Financial_Models as FM
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

Anfang = '18/06/2020'
Ende = '19/06/2020'

df1 = FM.GetDFPrices(['AGRO3', 'BBDC4', 'CYRE3', 'ENBR3', 'FLRY3', 'JBSS3'], [], Anfang, Ende, 'Close')
print(df1)
df2 = FM.GetDFPrices(['MDIA3', 'MGLU3', 'MRFG3', 'RLOG3', 'SLCE3', 'SUZB3'], [], Anfang, Ende, 'Close')
print(df2)
df3 = FM.GetDFPrices(['TIMP3', 'IVVB11'], ['BRL=X'], Anfang, Ende, 'Close')
print(df3)




