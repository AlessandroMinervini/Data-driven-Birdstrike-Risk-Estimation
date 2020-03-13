import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
matplotlib.verbose = False
from matplotlib import pyplot as plt


'Load CSV to compute BRI2, requirements: row, birdstrike, mensili, fauna'
path = 'BRI2_results/città/'

row_PSA = pd.read_csv(path + 'row.csv')
mensili_PSA = pd.read_csv(path + 'mensili.csv')
fauna = pd.read_csv('fauna.csv')
birdstrike_PSA = pd.read_csv(path + 'birdstrike_total.csv')

'Set the parameters'
groups = 17
n_months = 12
start = '2012' # First year where start the dataset
year = '2013' # First year where compute BRI2
limit = '2019-02-01 00:00:00' # Limit year of compute
years = int(limit[0:4]) - int(year) # Years of compute
data = '2013-02' # First month where compute BRI2

def compute_TNF(data, mensili, month, n_months, year):
    data = decrease_month(data)
    offset = int(start) # 2012
    first = int(year) #2013
    year = str(data)
    year = int(year[0:4])
    back_year = year - 1

    annual_TNF = 0
    monthly_TNF = 0

    months = {0: 'GENNAIO', 1: 'FEBBRAIO', 2: 'MARZO', 3: 'APRILE', 4: 'MAGGIO', 5: 'GIUGNO', 6: 'LUGLIO',
              7: 'AGOSTO', 8: 'SETTEMBRE', 9: 'OTTOBRE', 10: 'NOVEMBRE', 11: 'DICEMBRE'}

    # Annual TFN
    to_sum = year - offset
    #print(to_sum, 'to sum')
    for i in range(to_sum):
        tmp_mensili = mensili[mensili['anno'] == back_year]
        tmp_mensili = tmp_mensili['numeroMovimenti'].sum()
        annual_TNF  = annual_TNF + tmp_mensili
    annual_TNF = annual_TNF/to_sum

    # Monthly
    if month < n_months:
        i = month + 1
    else:
        i = 0
    y = 0
    tmp_mensili = mensili[(mensili['anno'] == year) | (mensili['anno'] == back_year)]
    while i < n_months:
        tmp_y = tmp_mensili[(tmp_mensili['anno'] == back_year)]
        tmp = tmp_y[tmp_y['mese'] == months[i]]
        tmp = tmp['numeroMovimenti'].sum()
        monthly_TNF = monthly_TNF + tmp
        i = i + 1
    while y <= month:
        tmp_y = tmp_mensili[(tmp_mensili['anno'] == year)]
        tmp = tmp_y[tmp_y['mese'] == months[y]]
        tmp = tmp['numeroMovimenti'].sum()
        monthly_TNF = monthly_TNF + tmp
        y = y + 1
    monthly_TNF = monthly_TNF / n_months
    return annual_TNF, monthly_TNF

def compute_DF(data, mensili, month, n_months):
    days = {0: 31, 1: 28, 2: 31, 3: 30, 4: 31, 5: 30, 6: 31,
              7: 31, 8: 30, 9: 31, 10: 30, 11: 31}
    months = {0: 'GENNAIO', 1: 'FEBBRAIO', 2: 'MARZO', 3: 'APRILE', 4: 'MAGGIO', 5: 'GIUGNO', 6: 'LUGLIO',
              7: 'AGOSTO', 8: 'SETTEMBRE', 9: 'OTTOBRE', 10: 'NOVEMBRE', 11: 'DICEMBRE'}

    global_DF = np.zeros([n_months])

    if month < n_months:
        i = month + 1
    else:
        i = 0
    y = 0
    index = 0
    data = decrease_month(data)
    year = str(data)
    year = int(year[0:4])
    back_year = year - 1
    tmp_mensili = mensili[(mensili['anno'] == year) | (mensili['anno'] == back_year)]
    # Prima parte dell'anno
    while i < n_months:
        tmp_y = tmp_mensili[(tmp_mensili['anno'] == back_year)]
        tmp = tmp_y[tmp_y['mese'] == months[i]]
        tmp = tmp['numeroMovimenti'].sum()
        global_DF[index] = tmp / days[i]
        i = i + 1
        index = index + 1
    while y <= month:
        tmp_y = tmp_mensili[(tmp_mensili['anno'] == year)]
        tmp = tmp_y[tmp_y['mese'] == months[y]]
        tmp = tmp['numeroMovimenti'].sum()
        global_DF[index] = tmp / days[y]
        y = y + 1
        index = index + 1
    return  global_DF

def compute_DB(row, data, month, n_months):
    data = str(data)
    data = data[0:7]
    DB = np.zeros([n_months, groups])
    data = data + '-01'

    year = str(int(data[0:4]) - 1)
    month = data[5:7]
    data = year + '-' + month + '-01'
    start, end = set_data(data)
    check_month = start.month
    daily = []
    m = 1
    observations = 1

    for i in range(groups):
        #print(i+1, '- gruppo')
        start, end = set_data(data)
        check_month = start.month
        daily = []
        m = 1
        observations = 1
        while m <= n_months:
            tmp = row[row['id_gruppo'] == float(i + 1)]
            tmp = (tmp[(tmp['ora'] >= str(start)) & (tmp['ora'] < str(end))])
            observations = observations + tmp.shape[0]
            nU = tmp['numeroUccelli'].sum()
            daily.append(nU)
            start, end = increase_data(start)
            if start.month != check_month:
                check_month = start.month
                DB[m-1, i] = np.sum(daily)/observations
                m = m + 1
                daily = []
                observations = 1
    return DB


def set_data(data):
    data = datetime.strptime(data, "%Y-%m-%d")
    start = data + timedelta(days=0)
    end = data + timedelta(days=1)
    return start, end

def increase_data(data):
    data = str(data)
    data = data[0:10]
    data = datetime.strptime(data, "%Y-%m-%d")
    start = data + timedelta(days=1)
    end = data + timedelta(days=2)
    return start, end

def compute_W(row, groups, fauna):
    W = np.zeros([groups])
    peso = 0
    for i in range(groups):
        tmp_row = row[row['id_gruppo'] == float(i + 1)]
        species = tmp_row['specie'].value_counts().index
        for ind in range(len(species)):
            tmp_f = fauna[fauna['specie'] == species[ind]]
            if tmp_f.shape[0] > 1:
                tmp_f = np.array(tmp_f)
                tmp_f = np.delete(tmp_f, range(1, tmp_f.shape[0]), 0)
                tmp_f = pd.DataFrame(tmp_f)
                tmp_f.columns = ['id', 'specie', 'famiglia', 'ordine', 'lunghezza', 'aperturaAlare', 'peso',
                                    'nomeLatino',
                                    'fileImmagine', 'frs', 'frp', 'fra', 'nomeInglese', 'euring', 'fenologia', 'note', 'id_famiglia',
                                 'uccello', 'visibile', 'hazardLevel', 'id_gruppo', 'valueID']
            peso = peso + tmp_f['peso'].sum()
        if len(species) == 0:
            W[i] = peso / 1
        else:
            W[i] = peso / len(species)
        peso = 0
    return W


def compute_AG(row_PSA, groups):
    AG = np.zeros([groups])
    for i in range(groups):
        tmp_row = row_PSA[row_PSA['id_gruppo'] == float(i + 1)]
        nU = tmp_row['numeroUccelli'].sum()
        if nU > 0:
            AG[i] = nU / tmp_row.shape[0]
        else:
            AG[i] = 1
    return AG

def compute_BS(groups, birdstrike, data, start, alt):
    if not alt:
        global_BS = np.zeros([groups])
        for i in range(groups):
            tmp_birdstrike = birdstrike[birdstrike['id_gruppo'] == float(i+1)]
            global_BS[i] = tmp_birdstrike.shape[0]
        return global_BS
    else:
        birdstrike = birdstrike_PSA
        global_BS = np.zeros([groups])
        year = int(data.year)
        y_to_count = year - int(start[0:4])
        for y in range(y_to_count):
            for i in range(groups):
                tmp_birdstrike = (birdstrike[(birdstrike['data'] >= str(year - 1)) & (birdstrike['data'] <= str(year))])
                tmp_birdstrike = tmp_birdstrike[tmp_birdstrike['id_gruppo'] == float(i+1)]
                global_BS[i] = global_BS[i] + tmp_birdstrike.shape[0]
            year = year - 1
        global_BS = global_BS / y_to_count
        return global_BS


def compute_EOF(birdstrike, groups):
    percentile = np.zeros([groups])
    for i in range(groups):
        eofs = []
        tmp_birdstrike = birdstrike[birdstrike['id_gruppo'] == float(i + 1)]
        if tmp_birdstrike.shape[0] != 0:
            tmp = np.array(tmp_birdstrike['A_effettoSulVolo'])
            print('DAL DB:', tmp)
            for j in range(tmp.shape[0]):
                if tmp[j] == 'Nessuno':
                    print('Nessuno')
                    eofs.append(1)
                elif tmp[j] == 'Atterraggio precauzionale':
                    print('Atterraggio')
                    eofs.append(3)
                elif tmp[j] == 'Decollo abortito':
                    print('Decollo')
                    eofs.append(2)
                elif tmp[j] == 'Altro (specificare)':
                    print('altro')
                    eofs.append(1)
                elif tmp[j] == 'Impedimento visivo':
                    print('impedimento')
                    eofs.append(4)
                else:
                    print('else')
                    eofs.append(1)
            print('eofs', eofs)
            perc = np.percentile(eofs, 95)
            print('percentile', perc)
            percentile[i] = perc
        else:
            percentile[i] = 1
    return percentile

def observations(row):
    for i in range(row.shape[0]):
        if i % 1 == 0:
            row.at[i, 'numeroUccelli'] = 2
            row.at[i, 'id_gruppo'] = 7.0
            row.at[i, 'peso'] = 310
    return row

def observations_months(rows):
    row = rows
    count1 = 0
    count2 = 0
    start_1 = '2014-01'
    end_1 = '2014-02'
    start_2 = '2018-01'
    end_2 = '2018-02'
    for i in range(row.shape[0]):
        if row.at[i, 'ora'] >= start_2 and row.at[i, 'ora'] < end_2:
            if i % 1 == 0:
                count1 = count1 +1
                row.at[i, 'numeroUccelli'] = 100
                row.at[i, 'id_gruppo'] = 16.0
                row.at[i, 'peso'] = 310
            else:
                count1 = count1 +1
                row.at[i, 'numeroUccelli'] = 100
                row.at[i, 'id_gruppo'] = 16.0
                row.at[i, 'peso'] = 310
        elif row.at[i, 'ora'] >= start_2 and row.at[i, 'ora'] < end_2:
            if i % 1 == 0:
                count2 = count2 + 1
                # row.at[i, 'numeroUccelli'] = 100
                # row.at[i, 'id_gruppo'] = 7.0
                # row.at[i, 'peso'] = 310
    print(count1, count2)
    #row.to_csv('PSA/testing_month/row_PSA.csv')
    return row

def BRI_plot(bri, bri2, two):
    BS = np.load('PSA/stats/count_BS_vers2.npy')
    labels = data_plot(start)
    X = []
    for i in range(len(bri)):
        X.append(i+1)

    fig, ax = plt.subplots()
    if not two:
        ax.plot(labels, bri) # X display no dates
        ax.plot(labels, BS) # X display no dates
        fig.autofmt_xdate()
        plt.gca().xaxis.set_major_locator(plt.LinearLocator(numticks=12))
        for i in range(n_months * years):
            if BS[i] > 0:
                plt.axvline(i, linewidth=0.3, color='r', linestyle='--')
    else:
        ax.plot(labels, bri) # X display no dates
        ax.plot(labels, bri2)
        fig.autofmt_xdate()
        plt.gca().xaxis.set_major_locator(plt.LinearLocator(numticks=12))
        for i in range(n_months * years):
            if BS[i] > 0:
                plt.axvline(i, linewidth=0.3, color='r', linestyle='--')

    ax.set(xlabel='Months', ylabel='BRI2',
           title='BRI2 scores from 2013 to 2018')
    ax.grid()

    #fig.savefig("PSA/bri_2013-18_vers2BS.png")
    plt.show()

def increase_month(data):
    data = str(data) + '-01'
    data = data[0:10]
    data = datetime.strptime(data, "%Y-%m-%d")
    month = data.month
    while data.month == month:
        data = data + timedelta(days=1)
    return data

def decrease_month(data):
    data = str(data) + '-01'
    data = data[0:10]
    data = datetime.strptime(data, "%Y-%m-%d")
    month = data.month
    while data.month == month:
        data = data - timedelta(days=1)
    while data.day != 1:
         data = data - timedelta(days=1)
    return data

def init_data(data):
    data = data + '-01'
    data = datetime.strptime(data, "%Y-%m-%d")
    data = data + timedelta(days=0)
    return data

def data_plot(start):
    data = start + '-12'
    datas = []
    for i in range(years * n_months):
        data = str(increase_month(data).date())
        d = str(data)
        datas.append(d[0:7])
    return datas

def count_BS(birdstrike):
    data = '2013-01'
    limit = '2019-01'
    data = init_data(data)
    annote_BS = []
    while str(data) < limit:
        print('---------------------------------')
        end = increase_month(data)
        tmp = birdstrike[(birdstrike['data'] >= str(data)) & (birdstrike['data'] <= str(end))]
        annote_BS.append(tmp.shape[0])
        data = increase_month(data)
    annote_BS = np.array(annote_BS)
    np.save(path + 'count_BS.npy', annote_BS)
    annote_BS[annote_BS>0] = 1
    #np.save('PSA/stats/count_BS_vers2_binary.npy', annote_BS)
    return annote_BS


def count_obs(row_PSA, groups, n_months, years, plot = False):
    data = '2013-01'
    limit = '2019-01'
    #data = init_data(data)
    annote_obs = []
    gropus_obs = np.zeros([groups, n_months*years])
    month = 0
    for g in range(groups):
        data = '2013-01'
        data = init_data(data)
        while str(data) < limit:
            print('---------------------------------')
            end = increase_month(data)
            tmp = row_PSA[row_PSA['id_gruppo'] == float(g + 1)]
            print(g)
            tmp = tmp[(tmp['ora'] >= str(data)) & (tmp['ora'] <= str(end))]
            gropus_obs[g,month] = tmp.shape[0]
            data = increase_month(data)
            month = month + 1
        month = 0
    #annote_obs = np.array(annote_obs)
    #annote_BS[annote_BS>0] = 1
    np.save('PSA/stats/count_OBS_groups.npy', gropus_obs)
    if plot:
        X = np.zeros([years*n_months])
        for i in range(years*n_months):
            X[i] = i

        fig, ax = plt.subplots()
        for g in range(groups):

            label = str(g + 1)
            ax.plot(range(years*n_months), gropus_obs[g,:], label=label)

        ax.set(xlabel='Months', ylabel='# brids',
               title='# birds from 2013 to 2018')
        ax.grid()
        leg = ax.legend()
        plt.show()

    to_hist = gropus_obs[:,42:46]
    to_hist = np.sum(to_hist, axis=1)
    X = []
    for i in range(groups):
        X.append(i + 1)
    plt.bar(X, to_hist, align='center', width=0.7, tick_label = X)
    plt.title(label='# birds from 42 to 45 month')
    plt.xlabel(xlabel='Groups')
    plt.ylabel(ylabel='# bird')
    plt.grid()
    plt.show()
    return gropus_obs

def count_BS_groups(birdstrike, plot = False):
    data = '2013-01'
    limit = '2019-01'
    BS_groups = np.zeros([groups])
    tmp = birdstrike[(birdstrike['data'] >= data) & (birdstrike['data'] <= limit)]
    for g in range(groups):
        tmp_birdstrike = tmp[tmp['id_gruppo'] == float(g + 1)]
        BS_groups[g] = tmp_birdstrike.shape[0]
    np.save('PSA/stats/count_BS_groups.npy', BS_groups)
    if plot:
        X = []
        for i in range(groups):
            X.append(i + 1)
        plt.bar(X, BS_groups, align='center', width=0.7, tick_label = X)
        plt.title(label='Birdstrike')
        plt.xlabel(xlabel='Groups')
        plt.ylabel(ylabel='# birdstrikes')
        plt.grid()
        plt.show()
    return BS_groups

def  merge_BS():
    birdstrike_A_PSA = pd.read_csv('birdstrike_A_PSA.csv')
    birdstrike_B_PSA = pd.read_csv('birdstrike_B_PSA.csv')
    birdstrike_C_PSA = pd.read_csv('birdstrike_C_PSA.csv')
    birdstrike_D_PSA = pd.read_csv('birdstrike_D_PSA.csv')

    birdstrike_A_PSA.columns = ['', 'id_aeroporto', 'data', 'data_', 'A_effettoSulVolo', 'id_specie_volatile', 'id_gruppo']
    birdstrike_B_PSA.columns = ['', 'id_aeroporto', 'data', 'data_', 'A_effettoSulVolo', 'id_specie_volatile', 'id_gruppo']
    birdstrike_C_PSA.columns = ['', 'id_aeroporto', 'data', 'data_', 'A_effettoSulVolo', 'id_specie_volatile', 'id_gruppo']
    birdstrike_D_PSA.columns = ['', 'id_aeroporto', 'data', 'data_', 'A_effettoSulVolo', 'id_specie_volatile', 'id_gruppo']

    birdstrike = pd.concat([birdstrike_D_PSA, birdstrike_A_PSA, birdstrike_B_PSA, birdstrike_C_PSA], axis=0)
    birdstrike = birdstrike.drop_duplicates(subset = ['data'])
    birdstrike.to_csv('birdstrike_PSA_total.csv')
    return birdstrike


def BRI2(row_PSA, years, fauna, n_months, mensili, birdstrike, groups, data):
    BRI2i = []

    data = init_data(data)
    #row_PSA = observations(row_PSA)
    #row_PSA = observations_months(row_PSA)

    # For stats.
    AGi = np.zeros([years*n_months, groups])
    Wi = np.zeros([years*n_months, groups])
    BSi = np.zeros([years*n_months, groups])
    EOFi = np.zeros([years*n_months, groups])
    GFi = np.zeros([years*n_months, groups])

    index = 0
    alt = False

    while str(data) < limit:
        #print('---------', data.year, '---------')
        print(data.year, 'ANNO-------------------------------------------------------------')
        for m in range(n_months):
            #print(data)
            print(m + 1, 'mese', 'anno', data.year)
            # Datasets
            row = row_PSA[(row_PSA['ora'] >= start) & (row_PSA['ora'] <= str(data))]
            tmp_birdstrike = (birdstrike[(birdstrike['data'] >= start) & (birdstrike['data'] <= str(data))])

            # Functions for parameters
            AG = compute_AG(row, groups)
            print('---------AG---------')
            #print(AG)
            W = compute_W(row_PSA, groups, fauna)
            print('---------W---------')
            #print(W)
            BS = compute_BS(groups, tmp_birdstrike, data, start, alt)
            print('---------BS---------')
            #print(BS)
            DF = compute_DF(data, mensili, m, n_months)
            print('---------DF---------')
            #print(DF)
            a_TNF, m_TNF = compute_TNF(data, mensili, m, n_months, year)
            print('---------A TFN---------')
            #print(a_TNF)
            print('---------M TFN---------')
            #print(m_TNF)
            EOF = compute_EOF(tmp_birdstrike, groups)
            print('---------EOF---------')
            print('EOF', EOF)
            DB = compute_DB(row, data, m, n_months)
            print('---------DB---------')
            #print(DB)

            # Compute BRI2
            # Fattore di gruppo
            GF = np.zeros([groups])
            for g in range(groups):
                GF[g] = W[g] * AG[g] * (BS[g] / a_TNF) * EOF[g]

            # Standardizzazione del Fattore Gruppo e calcolo del Fattore di Rischio
            sum_GF = np.sum(GF)
            GSR = np.zeros([n_months, groups])
            for z in range(n_months):
                for i in range(groups):
                    GSR[z, i] = (GF[i] / sum_GF) * DB[z, i]

            # Birdstrike Risk Index ver. 2
            sum_GSR = np.zeros([n_months])
            BRI2 = np.zeros([n_months])
            tmp_GSR = 0
            for z in range(n_months):
                for i in range(groups):
                    tmp_GSR = tmp_GSR + GSR[z, i]  # + GSR[y, m, i+1]
                BRI2[z] = (tmp_GSR * DF[z]) / m_TNF
                tmp_GSR = 0

            BRI2i.append(BRI2[-1])

            AGi[index,:] = AG
            Wi[index,:] = W
            BSi[index,:] = BS
            EOFi[index,:] = EOF
            GFi[index,:] = GF
            #DBi[index, :,:] = DB
            index = index + 1

            data = increase_month(data)
            # start = increase_month(start)
            # start = str(start)
            #print(BRI2)

    np.save(path +'AGi_test_2.npy', AGi)
    np.save(path +'Wi_test.npy', Wi)
    np.save(path +'BSi_test.npy', BSi)
    np.save(path +'EOFi_test.npy', EOFi)
    np.save(path +'GFi_test.npy', GFi)

    return BRI2i



''' ----------------------------------------------- TEST ------------------------------------------------------------'''
print('Test start:')
bri2 = BRI2(row_PSA, years, fauna, n_months, mensili_PSA, birdstrike_PSA, groups, data)
#np.save(path + 'bri_2013-18.npy', bri2)

BS = count_BS(birdstrike_PSA)

bri_old = []
BRI_plot(bri2, BS, two = False)




