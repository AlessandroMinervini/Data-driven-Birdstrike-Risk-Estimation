import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
matplotlib.verbose = False
from matplotlib import pyplot as plt


'''Load CSV to compute BRI2, requirements: mensili, birdstrikes, scheda'''
mensili_PSA = pd.read_csv('WRI/mensili_PSA.csv')
birdstrike_PSA = pd.read_csv('WRI/birdstrike_PSA_total.csv')
scheda_PSA = pd.read_csv('WRI/scheda_PSA.csv')

'''Set the parameters'''
n_months = 12
year = '2013' # First year where compute WRI
limit = '2019-02-01 00:00:00' # Limit year of compute
years = int(limit[0:4]) - int(year) # Years of compute
data = '2013-01' # First month where compute WRI


def compute_C1(data, mensili, birdstrike):
    ''' Setting dates'''
    next_data = str(increase_month(data))
    year = data.year
    month = str(data.month)

    const_C1 = 1250
    limit_BS = 20
    range_mov = 1000
    bound = 25

    months = {'1': 'GENNAIO', '2': 'FEBBRAIO', '3': 'MARZO', '4': 'APRILE', '5': 'MAGGIO', '6': 'GIUGNO', '7': 'LUGLIO',
              '8': 'AGOSTO', '9': 'SETTEMBRE', '10': 'OTTOBRE', '11': 'NOVEMBRE', '12': 'DICEMBRE'}

    ''' Compute mov/month'''
    tmp_mensili = mensili[mensili['anno'] == year]
    mov_mese = tmp_mensili[tmp_mensili['mese'] == months[month]]
    mov_mese = np.array(mov_mese['numeroMovimenti'])

    ''' Compute BS/month'''
    tmp_birdstrike = (birdstrike[(birdstrike['data'] >= str(data)) & (birdstrike['data'] <= next_data)])
    n_BS = tmp_birdstrike.shape[0]

    if n_BS < limit_BS:# and mov_mese <= range_mov:
        C1 = (n_BS * const_C1) / mov_mese
    else:
        C1 = bound
    return C1

def compute_C2(data, birdstrike):
    ''' Setting dates '''
    next_data = increase_month(data)

    fauna_table = {'27' : 25, '2' : 20, '1' : 20, '1440' : 20, '3' : 20, '34' : 20, '7' : 15, '1121' : 15, '38': 15,
                   '36' : 15, '1447' : 15, '1253' : 10, '14' : 5, '13' : 5, '15' : 5, '25' : 10, '64':5,
                   '12' : 5, '995' : 5, '40' : 5, '476' : 5}
    av_risk = 5
    C2 = 0

    ''' Compute C2 '''
    tmp_birdstrike = (birdstrike[(birdstrike['data'] >= str(data)) & (birdstrike['data'] <= str(next_data))])
    ids_specie = np.array(tmp_birdstrike['id_specie_volatile'])

    for i in range(ids_specie.shape[0]):
        specie = ids_specie[i]
        try:
            C2 = C2 + fauna_table[str(specie)]
        except:
            #print('Not in table')
            C2 = C2 + av_risk

    return C2

def compute_C3(data, birdstrike):
    ''' Setting dates '''
    next_data = increase_month(data)

    C3 = 0
    values = [1, 2, 3, 5]

    tmp_birdstrike = (birdstrike[(birdstrike['data'] >= str(data)) & (birdstrike['data'] <= str(next_data))])
    n_birds = np.array(tmp_birdstrike['coinvolti'])

    ''' Compute C3'''
    for i in range(n_birds.shape[0]):
        if n_birds[i] <= 1:
            C3 = C3 + values[0]
        elif n_birds[i] > 1 and n_birds[i] <= 10:
            C3 = C3 + values[1]
        elif n_birds[i] > 11 and n_birds[i] <= 100:
            C3 = C3 + values[2]
        elif n_birds[i] > 100:
            C3 = C3 + values[3]
        elif np.isnan(n_birds[i]):
            C3 = C3 + values[0]
    return C3

def compute_C4(data, birdstrike):
    ''' Setting dates '''
    next_data = increase_month(data)

    C4 = 0
    EOF_values = [4.3, 5, 6, 7.5, 10, 15]

    tmp_birdstrike = (birdstrike[(birdstrike['data'] >= str(data)) & (birdstrike['data'] <= str(next_data))])
    EOF = np.array(tmp_birdstrike['effettoSulVolo'])
    parti_colpite = np.array(tmp_birdstrike['parti_colpite'])

    ''' Compute C4'''
    for i in range(EOF.shape[0]):
        if EOF[i] == 'Nessuno':
            if pd.isnull(parti_colpite[i]):
                C4 = C4 + EOF_values[0]
            else:
                C4 = C4 + (EOF_values[0] * 2)
        elif EOF[i] == 'Decollo abortito' or EOF[i] == 'Ritardo':
            if pd.isnull(parti_colpite[i]):
                C4 = C4 + EOF_values[1]
            else:
                C4 = C4 + (EOF_values[1] * 2)
        elif EOF[i] == 'Atterraggio precauzionale':
            if pd.isnull(parti_colpite[i]):
                C4 = C4 + EOF_values[2]
            else:
                C4 = C4 + (EOF_values[2] * 2)
        elif  EOF[i] == 'Arresto motore':
            if pd.isnull(parti_colpite[i]):
                C4 = C4 + EOF_values[3]
            else:
                C4 = C4 + (EOF_values[3] * 2)
        elif EOF[i] == 'Atterraggio forzato':
            if pd.isnull(parti_colpite[i]):
                C4 = C4 + EOF_values[4]
            else:
                C4 = C4 + (EOF_values[4] * 2)
        elif EOF[i] == 'Impedimento visivo':
            if pd.isnull(parti_colpite[i]):
                C4 = C4 + EOF_values[5]
            else:
                C4 = C4 + (EOF_values[5] * 2)
        elif np.isnan(EOF[i]):
            if pd.isnull(parti_colpite[i]):
                C4 = C4 + EOF_values[0]
            else:
                C4 = C4 + (EOF_values[0] * 2)
    return C4

def compute_C5(data, scheda):
    ''' Setting dates '''
    next_data = increase_month(data)

    const_isp = 60
    const_value = 15
    bound = 15

    isp = (scheda[(scheda['data'] >= str(data)) & (scheda['data'] <= str(next_data))])

    ''' Compute C5'''
    non_programmate = (isp[isp['causaIspezione'] != 'Ispezione normale o programmata']).shape[0]
    if non_programmate < const_isp:
        C5 = (non_programmate*const_value) / const_isp
    else:
        C5 = bound
    return C5

def increase_month(data):
    data = str(data) + '-01'
    data = data[0:10]
    data = datetime.strptime(data, "%Y-%m-%d")
    month = data.month
    while data.month == month:
        data = data + timedelta(days=1)
    return data

def init_data(data):
    data = data + '-01'
    data = datetime.strptime(data, "%Y-%m-%d")
    data = data + timedelta(days=0)
    return data

def data_plot(year):
    start = str(int(year) - 1)
    data = start + '-12'
    datas = []
    for i in range(years * n_months):
        data = str(increase_month(data).date())
        d = str(data)
        datas.append(d[0:7])
    return datas

def WRI_plot(wri):
    '''Plot WRI scores'''
    BS = np.load('WRI/count_BS.npy')
    labels = data_plot(year)

    fig, ax = plt.subplots()
    ax.plot(labels, wri)
    fig.autofmt_xdate()
    plt.gca().xaxis.set_major_locator(plt.LinearLocator(numticks=12))
    for i in range(n_months * years):
        if BS[i] > 0:
            plt.axvline(i, linewidth=0.3, color='r', linestyle='--')

    ax.set(xlabel='Months', ylabel='WRI',
           title='WRI scores from 2013 to 2018')
    ax.grid()

    fig.savefig("WRI/wri_2013-18.png")
    plt.show()

def WRI(data, mensili, birdstrike, scheda):
    ''' Setting dates '''
    data = init_data(data)

    ''' Variables'''
    WRI_scores = []

    for i in range(years*n_months):
        C1 = compute_C1(data, mensili_PSA, birdstrike_PSA)
        C2 = compute_C2(data, birdstrike_PSA)
        C3 = compute_C3(data, birdstrike_PSA)
        C4 = compute_C4(data, birdstrike_PSA)
        C5 = compute_C5(data, scheda_PSA)

        WRI = (C1 + C2 + C3 + C4 + C5)/100
        WRI_scores.append(WRI)
        data = increase_month(data)

    return WRI_scores


WRI_s = WRI(data, mensili_PSA, birdstrike_PSA, scheda_PSA)
WRI_plot(WRI_s)