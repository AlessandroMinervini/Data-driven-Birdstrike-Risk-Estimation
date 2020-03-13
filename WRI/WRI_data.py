import numpy as np
import pandas as pd
import sqlalchemy as sql

connect_string = 'mysql://root:birdcontrol@localhost:3306/birdcontrol'

sql_engine = sql.create_engine(connect_string)

''' Id airports'''
airport_pisa = 7
airport_firenze = 5
airport_roma = 26
airport_milano = 12
airport_venezia = 84
airport_bologna = 8
airport_napoli = 37

''' Query on DB'''

query = '''
SELECT id_aeroporto, ora, numeroUccelli, riga_scheda.id_specie, specie,
	nomeInglese, fauna.id_gruppo, fauna.peso
        from birdcontrol.riga_scheda join birdcontrol.scheda on riga_scheda.id_scheda = scheda.id
	  			     join birdcontrol.fauna on riga_scheda.id_specie = fauna.id
where ora >= '2011-01-01 00:00:00'
'''

query_birdstrike_A = '''
SELECT id_aeroporto, bird_strikes.data, bird_strikes.A_data, bird_strikes.A_effettoSulVolo, bird_strikes.id_specie_volatile_a,
    fauna.id_gruppo, A_numeroVolatiliColpiti, A_partiColpite
        from birdcontrol.bird_strikes join birdcontrol.fauna on bird_strikes.id_specie_volatile_a = fauna.id
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_birdstrike_B = '''
SELECT id_aeroporto, bird_strikes.data, bird_strikes.B_data, bird_strikes.A_effettoSulVolo, bird_strikes.id_specie_volatile_b,
    fauna.id_gruppo, B_partiColpite
        from birdcontrol.bird_strikes join birdcontrol.fauna on bird_strikes.id_specie_volatile_b = fauna.id
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_birdstrike_C = '''
SELECT id_aeroporto, bird_strikes.data, bird_strikes.C_data, bird_strikes.A_effettoSulVolo, bird_strikes.id_specie_volatile_c,
    fauna.id_gruppo, C_numeroVolatiliColpiti, C_partiColpite
        from birdcontrol.bird_strikes join birdcontrol.fauna on bird_strikes.id_specie_volatile_c = fauna.id
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_birdstrike_D = '''
SELECT id_aeroporto, bird_strikes.data, bird_strikes.D_data, bird_strikes.A_effettoSulVolo, bird_strikes.id_specie_volatile_d,
    fauna.id_gruppo, D_numeroVolatili
        from birdcontrol.bird_strikes join birdcontrol.fauna on bird_strikes.id_specie_volatile_d = fauna.id
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_mensili = '''
SELECT id_aeroporto, mese, anno, impattiTotali, impattiConDanni, conEffettiSulVolo, numeroMovimenti
        from birdcontrol.movimenti_mensili
where anno >= '2011'
'''

query_scheda = '''
SELECT scheda.id_aeroporto, scheda.data, scheda.causaIspezione, scheda.causaIspezioneAltro
        from birdcontrol.scheda
where data >= '2011-01-01 00:00:00'
'''

# Get all since 2011.
birdstrike_A = pd.read_sql_query(query_birdstrike_A, sql_engine)
birdstrike_B = pd.read_sql_query(query_birdstrike_B, sql_engine)
birdstrike_C = pd.read_sql_query(query_birdstrike_C, sql_engine)
birdstrike_D = pd.read_sql_query(query_birdstrike_D, sql_engine)
mensili = pd.read_sql_query(query_mensili, sql_engine)
scheda = pd.read_sql_query(query_scheda, sql_engine)

birdstrike_A_PSA = birdstrike_A[birdstrike_A['id_aeroporto'] == airport_pisa]
birdstrike_B_PSA = birdstrike_B[birdstrike_B['id_aeroporto'] == airport_pisa]
birdstrike_C_PSA = birdstrike_C[birdstrike_C['id_aeroporto'] == airport_pisa]
birdstrike_D_PSA = birdstrike_D[birdstrike_D['id_aeroporto'] == airport_pisa]
mensili_PSA = mensili[mensili['id_aeroporto'] == airport_pisa]
scheda_PSA = scheda[scheda['id_aeroporto'] == airport_pisa]


# Merge Birdstrikes
birdstrike_A_PSA = birdstrike_A_PSA.drop(birdstrike_A_PSA.columns[0], axis=1)
birdstrike_B_PSA = birdstrike_B_PSA.drop(birdstrike_B_PSA.columns[0], axis=1)
birdstrike_C_PSA = birdstrike_C_PSA.drop(birdstrike_C_PSA.columns[0], axis=1)
birdstrike_D_PSA = birdstrike_D_PSA.drop(birdstrike_D_PSA.columns[0], axis=1)

cols = ['id_aeroporto', 'data', 'data_', 'effettoSulVolo', 'id_specie_volatile', 'id_gruppo', 'coinvolti',
        'parti_colpite']

birdstrike_B_PSA['coinvolti'] = 1
birdstrike_D_PSA['parti_colpite'] = ''

birdstrike_A_PSA.columns = ['id_aeroporto', 'data', 'data_', 'effettoSulVolo', 'id_specie_volatile', 'id_gruppo',
                            'coinvolti', 'parti_colpite']
birdstrike_B_PSA.columns = ['id_aeroporto', 'data', 'data_', 'effettoSulVolo', 'id_specie_volatile', 'id_gruppo',
                            'parti_colpite', 'coinvolti']
birdstrike_C_PSA.columns = ['id_aeroporto', 'data', 'data_', 'effettoSulVolo', 'id_specie_volatile', 'id_gruppo',
                            'coinvolti', 'parti_colpite']
birdstrike_D_PSA.columns = ['id_aeroporto', 'data', 'data_', 'effettoSulVolo', 'id_specie_volatile', 'id_gruppo',
                            'coinvolti', 'parti_colpite']

birdstrike_B_PSA = birdstrike_B_PSA[cols]

birdstrike = pd.concat([birdstrike_D_PSA, birdstrike_A_PSA, birdstrike_B_PSA, birdstrike_C_PSA], axis=0)
birdstrike = birdstrike.drop_duplicates(subset=['data'])

# Save data
mensili_PSA.to_csv('WRI/mensili_PSA.csv')
scheda_PSA.to_csv('WRI/scheda_PSA.csv')
birdstrike.to_csv('WRI/birdstrike_PSA_total.csv')
