import numpy as np
import pandas as pd
import sqlalchemy as sql
from sklearn.preprocessing import MinMaxScaler, normalize
from datetime import datetime, timedelta
from calendar import monthrange
import matplotlib
matplotlib.verbose = False
from matplotlib import pyplot as plt
import itertools
from PyAstronomy import pyasl

connect_string = 'mysql://root:birdcontrol@localhost:3306/birdcontrol'
#path = 'dataset/test/'
path = 'dataset/NAME-AIRPORT/'

sql_engine = sql.create_engine(connect_string)

#Selected Airport
#airport = 7 #Pisa
#airport = 5 #Firenze
#airport = 12 #Malpensa
#airport = 28 #Ciampino
#airport = 4 #Palermo
#airport = 84 #Venezia
#airport = 37 #Napoli
#airport = 8 #Bologna
#airport = 11 #Linate
#airport = 43 #Brescia
#airport = 3 #Alghero
#airport = 56 #Rimini
#airport = 16 #Catania
#airport = 44 #Verona
#airport = 26 #Fiumicino
#airport = 65 #Genova

# Config dates
start_date = '2012-01'
end_date = '2019-01'
# BS back1year
sd = '2011-01'
ed = '2018-01'

days = 365
years = int(end_date[0:4]) - int(start_date[0:4])
months = 12

# Config dataset filter
max_temp = 50.0
min_temp = -30.0
max_wind_i = 180.0
max_wind_d = 360.0
d_value = 999999.999


# Query on DB
query = '''
SELECT id_aeroporto, ora, numeroUccelli, riga_scheda.id_specie, riga_scheda.ambiente, suoloBagnato,
	meteo, temperatura, intensitaVento, direzioneVento, riga_scheda.distanzaDallaPista, riga_scheda.latitudineRilevamento,
	riga_scheda.longitudineRilevamento
        from birdcontrol.riga_scheda join birdcontrol.scheda on riga_scheda.id_scheda = scheda.id
	  			     join birdcontrol.fauna on riga_scheda.id_specie = fauna.id
where ora >= '2011-01-01 00:00:00'
'''

query_birdstrike_A = '''
SELECT id_aeroporto, bird_strikes.data
        from birdcontrol.bird_strikes
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_birdstrike_B = '''
SELECT id_aeroporto, bird_strikes.data
        from birdcontrol.bird_strikes
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_birdstrike_C = '''
SELECT id_aeroporto, bird_strikes.data
        from birdcontrol.bird_strikes
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_birdstrike_D = '''
SELECT id_aeroporto, bird_strikes.data
        from birdcontrol.bird_strikes
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_mov = '''
SELECT id_aeroporto, movimenti_mensili.data, movimenti_mensili.numeroMovimenti
        from birdcontrol.movimenti_mensili
where movimenti_mensili.data >= '2011-01-01 00:00:00'
'''

dataset = pd.read_sql_query(query, sql_engine)

movimenti = pd.read_sql_query(query_mov, sql_engine)


# Clean and manage missing data

# Meteo
dataset = dataset[dataset.meteo.str.len()>0]

# Temperature
nan_idx = dataset[pd.isnull(dataset['temperatura'])].index
dataset = dataset.drop(nan_idx).reset_index()
col_temp = np.array(dataset['temperatura'])
new_col_temp = []
for t in col_temp:
    if isinstance(t, str):
        try:
            new_col_temp.append(float(t))
        except ValueError:
            new_col_temp.append(d_value)

    else:
        new_col_temp.append(t)
new_col_temp = np.array(new_col_temp)
dataset['temperatura'] = new_col_temp
high_temp_idx = dataset[(dataset['temperatura'] > max_temp) | (dataset['temperatura']< min_temp)].index
dataset = dataset.drop(high_temp_idx).reset_index()

# Wind intensity
nan_idx = dataset[pd.isnull(dataset['intensitaVento'])].index
dataset = dataset.drop(nan_idx).reset_index(drop=True)
col_i_wind = np.array(dataset['intensitaVento'])
new_col_iWind = []
for t in col_i_wind:
    if isinstance(t, str):
        try:
            new_col_iWind.append(float(t))
        except ValueError:
            new_col_iWind.append(d_value)

    else:
        new_col_iWind.append(t)
new_col_iWind = np.array(new_col_iWind)
dataset['intensitaVento'] = new_col_iWind
to_del_idx = dataset[dataset['intensitaVento'] == d_value].index
trashold_idx = dataset[dataset['intensitaVento'] > max_wind_i].index
dataset = dataset.drop(to_del_idx).reset_index(drop=True)
dataset = dataset.drop(trashold_idx).reset_index(drop=True)

# Wind direction
nan_idx = dataset[pd.isnull(dataset['direzioneVento'])].index
dataset = dataset.drop(nan_idx).reset_index(drop=True)
to_del_idx = dataset[dataset['direzioneVento'] > max_wind_d].index
dataset = dataset.drop(to_del_idx).reset_index(drop=True)

# Distanzna dalla pista
nan_idx = dataset[pd.isnull(dataset['distanzaDallaPista'])].index
dataset = dataset.drop(nan_idx).reset_index(drop=True)
dataset['distanzaDallaPista'] = dataset['distanzaDallaPista'].round(3)

# Latitudine
nan_idx = dataset[pd.isnull(dataset['latitudineRilevamento'])].index
dataset = dataset.drop(nan_idx).reset_index(drop=True)
dataset['latitudineRilevamento'] = dataset['latitudineRilevamento'].round(3)

# Longitudine
nan_idx = dataset[pd.isnull(dataset['longitudineRilevamento'])].index
dataset = dataset.drop(nan_idx).reset_index(drop=True)
dataset['longitudineRilevamento'] = dataset['longitudineRilevamento'].round(3)

# Ambiente
nan_idx = dataset[pd.isnull(dataset['ambiente'])].index
dataset = dataset.drop(nan_idx).reset_index(drop=True)
dataset['ambiente'] = dataset['ambiente'].str.lower()

# Suolo bagnato
nan_idx = dataset[pd.isnull(dataset['suoloBagnato'])].index
dataset = dataset.drop(nan_idx).reset_index(drop=True)

# Save indexes for species features
species = sorted(set(dataset['id_specie']))
enum = list(itertools.chain(*list(enumerate(species))))
enum = {enum[i]: enum[i + 1] for i in range(0, len(enum), 2)}
species_idx = {v: k for k, v in enum.items()}

# Save data for meteo features
meteo = dataset['meteo']
meteo = meteo.drop_duplicates()
meteo = list(itertools.chain(*list(enumerate(meteo))))
meteo = {meteo[i]: meteo[i + 1] for i in range(0, len(meteo), 2)}
meteo_idx = {v: k for k, v in meteo.items()}
dataset['meteo'] = dataset['meteo'].replace(meteo_idx)

# Save data for temp features
temp = sorted(set(dataset['temperatura']))
temp = list(itertools.chain(*list(enumerate(temp))))
temp = {temp[i]: temp[i + 1] for i in range(0, len(temp), 2)}
temp_idx = {v: k for k, v in temp.items()}

# Save data for wind intensity features
i_wind = sorted(set(dataset['intensitaVento']))
i_wind = list(itertools.chain(*list(enumerate(i_wind))))
i_wind = {i_wind[i]: i_wind[i + 1] for i in range(0, len(i_wind), 2)}
i_wind_idx = {v: k for k, v in i_wind.items()}

# Save data for wind direction features
d_wind = sorted(set(dataset['direzioneVento']))
d_wind = list(itertools.chain(*list(enumerate(d_wind))))
d_wind = {d_wind[i]: d_wind[i + 1] for i in range(0, len(d_wind), 2)}
d_wind_idx = {v: k for k, v in d_wind.items()}

# Save data for airport runway features
runway = sorted(set(dataset['distanzaDallaPista']))
runway = list(itertools.chain(*list(enumerate(runway))))
runway = {runway[i]: runway[i + 1] for i in range(0, len(runway), 2)}
runway_idx = {v: k for k, v in runway.items()}

# Save data for latitude features
latitude = sorted(set(dataset['latitudineRilevamento']))
latitude = list(itertools.chain(*list(enumerate(latitude))))
latitude = {latitude[i]: latitude[i + 1] for i in range(0, len(latitude), 2)}
latitude_idx = {v: k for k, v in latitude.items()}

# Save data for longitude features
longitude = sorted(set(dataset['longitudineRilevamento']))
longitude = list(itertools.chain(*list(enumerate(longitude))))
longitude = {longitude[i]: longitude[i + 1] for i in range(0, len(longitude), 2)}
longitude_idx = {v: k for k, v in longitude.items()}

# Save data for ambiente features
ambiente = sorted(set(dataset['ambiente']))
ambiente = list(itertools.chain(*list(enumerate(ambiente))))
ambiente = {ambiente[i]: ambiente[i + 1] for i in range(0, len(ambiente), 2)}
ambiente_idx = {v: k for k, v in ambiente.items()}
dataset['ambiente'] = dataset['ambiente'].replace(ambiente_idx)
print(ambiente_idx)
# Save data for suolo banganto features
suoloBagnato = sorted(set(dataset['suoloBagnato']))
suoloBagnato = list(itertools.chain(*list(enumerate(suoloBagnato))))
suoloBagnato = {suoloBagnato[i]: suoloBagnato[i + 1] for i in range(0, len(suoloBagnato), 2)}
suoloBagnato_idx = {v: k for k, v in suoloBagnato.items()}

#dataset.to_csv('global_dataset.csv')
dataset = dataset[dataset['id_aeroporto'] == airport]
movimenti = movimenti[movimenti['id_aeroporto'] == airport]
#dataset.to_csv(path + 'dataset_PSA.csv')
#np.save(path + 'dataset.npy', dataset)

birdstrike_A = pd.read_sql_query(query_birdstrike_A, sql_engine)
birdstrike_B = pd.read_sql_query(query_birdstrike_B, sql_engine)
birdstrike_C = pd.read_sql_query(query_birdstrike_C, sql_engine)
birdstrike_D = pd.read_sql_query(query_birdstrike_D, sql_engine)

birdstrike_A_PSA = birdstrike_A[birdstrike_A['id_aeroporto'] == airport]
birdstrike_B_PSA = birdstrike_B[birdstrike_B['id_aeroporto'] == airport]
birdstrike_C_PSA = birdstrike_C[birdstrike_C['id_aeroporto'] == airport]
birdstrike_D_PSA = birdstrike_D[birdstrike_D['id_aeroporto'] == airport]

# Merge Birdstrikes
birdstrike_A_PSA.columns = ['id_aeroporto', 'data']
birdstrike_B_PSA.columns = ['id_aeroporto', 'data']
birdstrike_C_PSA.columns = ['id_aeroporto', 'data']
birdstrike_D_PSA.columns = ['id_aeroporto', 'data']

birdstrike = pd.concat([birdstrike_D_PSA, birdstrike_A_PSA, birdstrike_B_PSA, birdstrike_C_PSA], axis=0)

labels = birdstrike.drop_duplicates(subset=['data'])
labels.to_csv(path + 'labels.csv')


# Making ground truth
def init_data(data):
    data = data + '-01'
    data = datetime.strptime(data, "%Y-%m-%d")
    data = data + timedelta(days=0)
    return data

def increase_day(data):
    data = str(data)
    data = data[0:10]
    data = datetime.strptime(data, "%Y-%m-%d")
    data = data + timedelta(days=1)
    return data

def increase_month(data):
    data = str(data) + '-01'
    data = data[0:10]
    data = datetime.strptime(data, "%Y-%m-%d")
    month = data.month
    while data.month == month:
        data = data + timedelta(days=1)
    return data

def compute_BS(split_size = 75):
    s_day = init_data(start_date)
    BS = []
    for i in range(years*days):
        e_day = increase_day(s_day)
        tmp = labels[(labels['data'] >= str(s_day)) & (labels['data'] < str(e_day))]
        BS.append(tmp.shape[0])
        s_day = increase_day(s_day)
    BS = np.array(BS)
    split_idx = round((BS.shape[0]*split_size)/100)
    train_BS = BS[0:split_idx]
    test_BS = BS[split_idx:]
    np.save(path + 'train_BS.npy', train_BS)
    np.save(path + 'test_BS.npy', test_BS)
    return BS

def BS_back1year(split_size = 75):
    s_day = init_data(sd)
    y = int(ed[0:4]) - int(sd[0:4])
    BS = []
    for i in range(y*days):
        e_day = increase_day(s_day)
        tmp = labels[(labels['data'] >= str(s_day)) & (labels['data'] < str(e_day))]
        BS.append(tmp.shape[0])
        s_day = increase_day(s_day)
    BS = np.array(BS)
    split_idx = round((BS.shape[0]*split_size)/100)
    train_BS = BS[0:split_idx]
    test_BS = BS[split_idx:]
    np.save(path + 'train_BS_back1year.npy', train_BS)
    np.save(path + 'test_BS_back1year.npy', test_BS)
    return BS

def compute_GT(win = 31, split_size = 75):
    s_day = init_data(start_date)
    GT = []
    for i in range(years*days):
        e_day = increase_day(s_day)
        tmp = labels[(labels['data'] >= str(s_day)) & (labels['data'] < str(e_day))]
        GT.append(tmp.shape[0])
        s_day = increase_day(s_day)
    GT = np.array(GT)
    # Smoothing
    GT = pyasl.smooth(GT, win, 'hanning')
    # Normalization
    GT = GT.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(GT)
    GT = scaler.transform(GT)
    # Splitting train/test
    split_idx = round((GT.shape[0]*split_size)/100)
    train_GT = GT[0:split_idx]
    test_GT = GT[split_idx:]
    np.save(path + 'train_GT.npy', train_GT)
    np.save(path + 'test_GT.npy', test_GT)
    np.save(path + 'GT.npy', GT)
    return train_GT, test_GT, GT

def compute_movements(split_size = 75):
    mov = []
    s_day = init_data(sd)
    for i in range(years * months):
        print(i, '------')
        days_in_month = monthrange(s_day.year, s_day.month)
        e_day = increase_month(s_day)
        tmp = movimenti[(movimenti['data'] >= (s_day.date())) & (movimenti['data'] < (e_day.date()))]
        n_movements = float(tmp['numeroMovimenti'] / 30)
        print(n_movements)
        for j in range(days_in_month[1]):
            mov.append(n_movements)
        s_day = increase_month(s_day)
    print(mov)
    mov = np.array(mov)
    split_idx = round((mov.shape[0]*split_size)/100)
    train_n_m = mov[0:split_idx]
    test_n_m = mov[split_idx:]
    np.save(path + 'train_n_m.npy', train_n_m)
    np.save(path + 'test_n_m.npy', test_n_m)
    return train_n_m, test_n_m


# Making daily time series for features
species_bins = len(species_idx)
meteo_bins = len(meteo_idx)
temperature_bins = int(len(temp_idx)/2)
i_wind_bins = int(len(i_wind_idx)/2)
d_wind_bins = int(len(d_wind_idx)/2)
d_runway_bins = int(len(runway_idx)/1000)
ambiente_bins = int(len(ambiente_idx)/2)
test_bins = int(len(runway_idx))

print(species_bins)
print(meteo_bins)
print(temperature_bins)
print(i_wind_bins)
print(d_wind_bins)
print(d_runway_bins)
print(ambiente_bins)


def build_features():
    s_day = init_data(start_date)

    species_features = np.zeros([years * days, species_bins])
    meteo_features = np.zeros([years * days, meteo_bins])
    temperature_features = np.zeros([years * days, temperature_bins])
    i_wind_features = np.zeros([years * days, i_wind_bins])
    d_wind_features = np.zeros([years * days, d_wind_bins])
    d_runway_features = np.zeros([years * days, d_runway_bins])
    ambiente_features = np.zeros([years * days, ambiente_bins])
    suoloBagnato_features = np.zeros([years * days])
    movements = np.zeros([years * days])


    for i in range(years * days):
        e_day = increase_day(s_day)
        tmp = dataset[(dataset['ora'] >= str(s_day)) & (dataset['ora'] < str(e_day))]

        # Species
        tmp_species = np.array(tmp[['numeroUccelli', 'id_specie']])
        for j in range(tmp_species.shape[0]):
            species_features[i,species_idx[tmp_species[j,1]]] = species_features[i,species_idx[tmp_species[j,1]]] + tmp_species[j,0]
        # Meteo
        tmp_meteo = np.array(tmp['meteo'])
        meteo_features[i,:], _ = np.histogram(tmp_meteo, bins=meteo_bins)
        # Temperature
        tmp_temp = np.array(tmp['temperatura'])
        temperature_features[i,:], _ = np.histogram(tmp_temp, bins=temperature_bins)
        # Wind intensity
        tmp_wind_i = np.array(tmp['intensitaVento'])
        i_wind_features[i,:], _ = np.histogram(tmp_wind_i, bins=i_wind_bins)
        # Wind direction
        tmp_wind_d = np.array(tmp['direzioneVento'])
        d_wind_features[i,:], _ = np.histogram(tmp_wind_d, bins=d_wind_bins)
        # Airport runway distance
        tmp_runway = np.array(tmp['distanzaDallaPista'])
        d_runway_features[i,:], _ = np.histogram(tmp_runway, bins=d_runway_bins)
        # Ambiente
        tmp_ambiente = np.array(tmp['ambiente'])
        ambiente_features[i,:], _ = np.histogram(tmp_ambiente, bins=ambiente_bins)
        # Suolo bagnato
        tmp_suoloBagnato = np.array(tmp['suoloBagnato'])
        suoloBagnato_features[i] = tmp_suoloBagnato.shape[0]

        s_day = increase_day(s_day)

    return species_features, meteo_features, temperature_features, i_wind_features, d_wind_features, d_runway_features,\
           ambiente_features, suoloBagnato_features

def split_features(s, m, t, i_w, d_w, d_r, am, sb, split_size = 75):
    size = s.shape[0]
    print(s.shape[1])
    print(m.shape[1])
    print(t.shape[1])
    print(i_w.shape[1])
    print(d_w.shape[1])
    print(d_r.shape[0])
    print(am.shape[0])

    split_idx = round((size*split_size)/100)

    train_s = s[0:split_idx,:]
    test_s = s[split_idx:,:]
    train_m = m[0:split_idx,:]
    test_m = m[split_idx:,:]
    train_t = t[0:split_idx,:]
    test_t = t[split_idx:,:]
    train_i_w = i_w[0:split_idx,:]
    test_i_w = i_w[split_idx:,:]
    train_d_w = d_w[0:split_idx,:]
    test_d_w = d_w[split_idx:,:]
    train_d_r = d_r[0:split_idx,:]
    test_d_r = d_r[split_idx:,:]
    train_am = am[0:split_idx,:]
    test_am = am[split_idx:,:]
    train_sb = sb[0:split_idx]
    test_sb = sb[split_idx:]

    np.save(path + 'train_s.npy', train_s)
    np.save(path + 'test_s.npy', test_s)
    np.save(path + 'train_m.npy', train_m)
    np.save(path + 'test_m.npy', test_m)
    np.save(path + 'train_t.npy', train_t)
    np.save(path + 'test_t.npy', test_t)
    np.save(path + 'train_i_w.npy', train_i_w)
    np.save(path + 'test_i_w.npy', test_i_w)
    np.save(path + 'train_d_w.npy', train_d_w)
    np.save(path + 'test_d_w.npy', test_d_w)
    np.save(path + 'train_d_r.npy', train_d_r)
    np.save(path + 'test_d_r.npy', test_d_r)
    np.save(path + 'train_am.npy', train_am)
    np.save(path + 'test_am.npy', test_am)
    np.save(path + 'train_sb.npy', train_sb)
    np.save(path + 'test_sb.npy', test_sb)

    return train_s, test_s, train_m, test_m, train_t, test_t, train_i_w, test_i_w, train_d_w, test_d_w, train_d_r,\
           test_d_r, train_am, test_am, train_sb, test_sb

def GT_plot(GT):
    GT = np.squeeze(np.array(GT))
    l = len(GT)
    X = []
    for i in range(l):
        X.append(i)
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    ax.plot(X[0:365], GT[0:365])
    ax.plot(X[0:365], GT[0:365])
    ax.set(xlabel='Days', ylabel='Ground Truth',
           title='Florence: Ground Truth from 2012 to 2013')
    ax.grid()
    plt.show()
    fig.savefig(path + "GT_1y.png")

def plot_feat(feat):
    fig, ax = plt.subplots(figsize=(10, 10))
    for day in range(feat.shape[0]):
        ax.plot(feat[day, :])
        ax.grid()
    plt.show()


s, m, t, i_w, d_w, d_r, am, sb = build_features()
compute_movements()
split_features(s, m, t, i_w, d_w, d_r, am, sb)
compute_BS()
BS_back1year()
train_GT, test_GT, GT = compute_GT()
GT_plot(GT)



