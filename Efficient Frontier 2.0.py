import Impactus_Financial_Models as FM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as SO

Selic = 0.0225
MinConcentration = 0 #Alocação mínima por ativo
MaxConcentration = 1 #Alocação máxima por ativo

bottomBorder = 0 #Retorno mínimo a ser mostrado nos gráficos
topBorder = 0.80 # Retorno máximo a ser mostrado nos gráficos

StartOfSim = '01/01/2010' # Início da série
EndOfSim = '' # Final da série ('') significa 'hoje'

Stocks = ['MGLU3', 'JBSS3', 'PETR3', 'OIBR3', 'CNTO3', 'BPAC11', 'LCAM3', 'AZUL4', 'SMTO3', 'FLRY3']

Indices = [] # Caso a carteira tenha ativos que não levam '.SA' no final. Ex: 'BRL=X'

Alocs = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10] # Posições atuais/ de uma carteira qualquer. Em ordem alfabética, começando por 'Indices'

Positions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # 1 para posição comprada, 0 para posição vendida

# Puxa os preços do Yahoo Finance e transforma em log retornos
df = FM.GetDFPrices(Stocks, Indices, StartOfSim, EndOfSim, 'Adj Close')
dfLog = np.log(df/df.shift(1))

for u in range(0, len(Stocks)):
    if Positions[u] == 0:
        dfLog.iloc[:, u] = -1 * dfLog.iloc[:, u]
    else:
        pass

# Tamanho do gráfico:
plt.figure(figsize=(12, 8))
############################################################################################
# Transforma os pesos do portfólio em porcentagens:
weightsPort = np.array(Alocs)/100

# Calcula os retornos esperados pra cada ação do portfólio atual:
retPort = np.sum(dfLog.mean()*weightsPort*252)

# print(dfLog.mean())
# print(weightsPort)
# print(retPort)

# Calcula a volatilidade do portfólio atual:
volPort = np.sqrt(np.dot(weightsPort.T, np.dot(dfLog.cov()*252, weightsPort)))

# Calcula o Sharpe do portfólio atual:
sharpePort = (retPort - (np.log(1 + Selic)-1))/volPort

# Retorna as ações e os pesos do portfólio atual:

Stocks = Indices + Stocks

data = {'Stocks': Stocks, 'Weights': weightsPort}
print('\nCurrent portfolio:\n')
print(pd.DataFrame(data))

#Retorna o retorno esperado, volatilidade e Sharpe do portfólio atual
l = [retPort, volPort, sharpePort]
m = ['Exp Return', 'Volatility', 'Shrp Ratio']
data = {'Data': m, 'Numbers': l}
print()
print(pd.DataFrame(data))

# Plota o portfólio atual:

plt.scatter(volPort, retPort, c='crimson', label='Current Portfolio')
############################################################################################
# # Essa função está desativada, mas pode ser usada pra plotar portfólios com pesos aleatórios.
# # O método antigo da Carteira era assim, mas deixava de funcionar bem com vários ativos.
#
# # Número de portfólios aleatórios (quanto maior, mais lento)
# num_ports = 6000
#
# # Cria matrizes de zeros que serão preenchidas adiante:
# ret_arr = np.zeros(num_ports)
# vol_arr = np.zeros(num_ports)
# sharpe_arr = np.zeros(num_ports)
#
# # Esse loop repete 'num_ports' vezes, e calcula cada portfólio com pesos aleatórios
# for x in range(num_ports):
#     # Calcula pesos aleatórios e normaliza (p/ ficar entre 0 e 1)
#     weights = np.array(np.random.random(len(Stocks)))
#     weights = weights/np.sum(weights)
#
#     # Calcula o retorno esperado do portfólio
#     ret_arr[x] = np.sum(dfLog.mean()*weights*252)
#
#     # Calcula a volatilidade do portfólio
#     vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(dfLog.cov()*252, weights)))
#
#     # Calcula o Sharpe do portfólio
#     sharpe_arr[x] = (ret_arr[x] - (np.log(1 + Selic)-1))/vol_arr[x]
#
# # Plota os portfólios aleatórios:
# plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
############################################################################################
# Calcula o retorno esperado, volatilidade e Sharpe dado uma matriz de pesos:
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(dfLog.mean()*weights)*252
    vol = np.sqrt(np.dot(weights.T, np.dot(dfLog.cov()*252, weights)))
    sr = (ret - (np.log(1 + Selic)-1))/vol
    return np.array([ret, vol, sr])

# Retorna 0 se a soma dos pesos for 1:
def check_sum(weights):
    return np.sum(weights)-1

############################################################################################
# limitações:
cons = ({'type': 'eq', 'fun': check_sum})

# Limite de concentração:
bounds = []

# Valores que o modelo começará testando:
init_guess = []

# Preenche as duas variáveis acima:
for stock in Stocks:
    bounds.append((MinConcentration, MaxConcentration))
    init_guess.append(1/len(Stocks))

bounds = tuple(bounds)

############################################################################################
# Calcula níveis de retorno esperado:
frontier_y = np.linspace(bottomBorder, topBorder, 200)

# Retorna a volatilidade da função get_ret_vol_sr:
def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]

############################################################################################
frontier_x = []

# Para cada nível de retorno, minimiza a volatilidade e coloca na lista 'frontier_x':
for possible_return in frontier_y:
    cons = ({'type': 'eq', 'fun': check_sum},
            {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})

    result = SO.minimize(minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    frontier_x.append(result['fun'])

# Cria um dicionário com a 'frontier_x' e a 'frontier_y' e transforma em dataframe:
data = {'Volatility': frontier_x, 'Expected Returns': frontier_y}
FEficiente = pd.DataFrame(data)

#Cria uma coluna com o Sharpe de cada ponto:
FEficiente['Sharpe Ratio'] = (FEficiente['Expected Returns'] - (np.log(1 + Selic) - 1)) / FEficiente['Volatility']

# Plota a fronteira eficiente:
plt.plot(frontier_x, frontier_y, 'dimgray', linewidth=2)

############################################################################################
# Calcula o portfólio de menor volatilidade
print('\nMin Volatility:\n')
cons = ({'type': 'eq', 'fun': check_sum},
        {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - FEficiente['Expected Returns'][FEficiente['Volatility'].argmin()]})

result = SO.minimize(minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
data = {'Stocks': Stocks, 'Weights': np.round(result.x, 2)}

print(pd.DataFrame(data))
print()

l = list(get_ret_vol_sr(result.x))
m = ['Exp Return', 'Volatility', 'Shrp Ratio']
data = {'Data': m, 'Numbers': l}

print(MinVol := pd.DataFrame(data))

# Plota as informações encontradas
plt.scatter(MinVol.iloc[1]['Numbers'], MinVol.iloc[0]['Numbers'], c='lime', label='Minimum Volatility')

############################################################################################
# Calcula o portfólio de maior volatilidade
print('\nMax Volatility:\n')
cons = ({'type': 'eq', 'fun': check_sum},
        {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - FEficiente['Expected Returns'][FEficiente['Volatility'].argmax()]})

result = SO.minimize(minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
data = {'Stocks': Stocks, 'Weights': np.round(result.x, 2)}

print(pd.DataFrame(data))
print()

l = list(get_ret_vol_sr(result.x))
m = ['Exp Return', 'Volatility', 'Shrp Ratio']
data = {'Data': m, 'Numbers': l}

print(MaxVol := pd.DataFrame(data))
plt.scatter(MaxVol.iloc[1]['Numbers'], MaxVol.iloc[0]['Numbers'], c='darkviolet', label='Maximum Volatility')

############################################################################################
# Calcula o portfólio de maior Sharpe
print('\nMax Sharpe:\n')
cons = ({'type': 'eq', 'fun': check_sum},
        {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - FEficiente['Expected Returns'][FEficiente['Sharpe Ratio'].argmax()]})

result = SO.minimize(minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
data = {'Stocks': Stocks, 'Weights': np.round(result.x, 2)}

print(pd.DataFrame(data))
print()

l = list(get_ret_vol_sr(result.x))
m = ['Exp Return', 'Volatility', 'Shrp Ratio']
data = {'Data': m, 'Numbers': l}

print(MaxShpe := pd.DataFrame(data))
plt.scatter(MaxShpe.iloc[1]['Numbers'], MaxShpe.iloc[0]['Numbers'], c='darkorange', label='Maximum Sharpe Ratio')

############################################################################################
# Estabelece informações para criar o gráfico
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.legend()
# plt.savefig('cover.png')
plt.show()

############################################################################################
print()
# Agora o usuário define um retorno esperado alvo:
targetRet = float(input('Type an expected return level: '))
print()

cons = ({'type': 'eq', 'fun': check_sum},
        {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - targetRet})

result = SO.minimize(minimize_volatility, x0=init_guess, method='SLSQP', bounds=bounds, constraints=cons)
data = {'Stocks': Stocks, 'Weights': np.round(result.x, 2)}

print(pd.DataFrame(data))
print()

l = list(get_ret_vol_sr(result.x))
m = ['Exp Return', 'Volatility', 'Shrp Ratio']
data = {'Data': m, 'Numbers': l}

print(target := pd.DataFrame(data))
############################################################################################
# Estabelece informações para criar o gráfico

plt.figure(figsize=(12, 8))
plt.plot(frontier_x, frontier_y, 'dimgray', linewidth=2)
plt.scatter(volPort, retPort, c='crimson', label='Current Portfolio')
plt.scatter(MinVol.iloc[1]['Numbers'], MinVol.iloc[0]['Numbers'], c='lime', label='Minimum Volatility')
plt.scatter(MaxVol.iloc[1]['Numbers'], MaxVol.iloc[0]['Numbers'], c='darkviolet', label='Maximum Volatility')
plt.scatter(MaxShpe.iloc[1]['Numbers'], MaxShpe.iloc[0]['Numbers'], c='darkorange', label='Maximum Sharpe Ratio')
plt.scatter(target.iloc[1]['Numbers'], target.iloc[0]['Numbers'], c='springgreen', label='Target Portfolio')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.legend()
# plt.savefig('cover.png')
plt.show()

############################################################################################

