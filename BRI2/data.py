import numpy as np
import pandas as pd
import sqlalchemy as sql

connect_string = 'mysql://root:birdcontrol@localhost:3306/birdcontrol'

sql_engine = sql.create_engine(connect_string)

#airport = 5  #Firenze
#airport = 12 #Malpensa
#airport = 8 #Bologna
#airport = 16 #Catania
#airport = 4 #Palermo
#airport = 7  #Pisa
#airport = 43 #Brescia
#airport = 28 #Ciampino
#airport = 44 #Verona
#airport = 44 #Verona
#airport = 11 #Linate
airport = 65 #Genova


path = 'BRI2_results/Genova/'


query = '''
SELECT id_aeroporto, ora, numeroUccelli, riga_scheda.id_specie, specie,
	nomeInglese, fauna.id_gruppo, fauna.peso
        from birdcontrol.riga_scheda join birdcontrol.scheda on riga_scheda.id_scheda = scheda.id
	  			     join birdcontrol.fauna on riga_scheda.id_specie = fauna.id
where ora >= '2011-01-01 00:00:00'
'''

query_birdstrike_A = '''
SELECT id_aeroporto, bird_strikes.data, bird_strikes.A_data, bird_strikes.A_effettoSulVolo, bird_strikes.id_specie_volatile_a,
    fauna.id_gruppo
        from birdcontrol.bird_strikes join birdcontrol.fauna on bird_strikes.id_specie_volatile_a = fauna.id
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_birdstrike_B = '''
SELECT id_aeroporto, bird_strikes.data, bird_strikes.B_data, bird_strikes.A_effettoSulVolo, bird_strikes.id_specie_volatile_b,
    fauna.id_gruppo
        from birdcontrol.bird_strikes join birdcontrol.fauna on bird_strikes.id_specie_volatile_b = fauna.id
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_birdstrike_C = '''
SELECT id_aeroporto, bird_strikes.data, bird_strikes.C_data, bird_strikes.A_effettoSulVolo, bird_strikes.id_specie_volatile_c,
    fauna.id_gruppo
        from birdcontrol.bird_strikes join birdcontrol.fauna on bird_strikes.id_specie_volatile_c = fauna.id
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_birdstrike_D = '''
SELECT id_aeroporto, bird_strikes.data, bird_strikes.D_data, bird_strikes.A_effettoSulVolo, bird_strikes.id_specie_volatile_d,
    fauna.id_gruppo
        from birdcontrol.bird_strikes join birdcontrol.fauna on bird_strikes.id_specie_volatile_d = fauna.id
where bird_strikes.data >= '2011-01-01 00:00:00'
'''

query_mensili = '''
SELECT id_aeroporto, mese, anno, impattiTotali, impattiConDanni, conEffettiSulVolo, numeroMovimenti
        from birdcontrol.movimenti_mensili
where anno >= '2011'
'''

# Get all scheda rows since 2011.
rows = pd.read_sql_query(query, sql_engine)
birdstrike_A = pd.read_sql_query(query_birdstrike_A, sql_engine)
birdstrike_B = pd.read_sql_query(query_birdstrike_B, sql_engine)
birdstrike_C = pd.read_sql_query(query_birdstrike_C, sql_engine)
birdstrike_D = pd.read_sql_query(query_birdstrike_D, sql_engine)
mensili = pd.read_sql_query(query_mensili, sql_engine)

rows = rows[rows['id_aeroporto'] == airport]
birdstrike_A = birdstrike_A[birdstrike_A['id_aeroporto'] == airport]
birdstrike_B = birdstrike_B[birdstrike_B['id_aeroporto'] == airport]
birdstrike_C = birdstrike_C[birdstrike_C['id_aeroporto'] == airport]
birdstrike_D = birdstrike_D[birdstrike_D['id_aeroporto'] == airport]
mensili = mensili[mensili['id_aeroporto'] == airport]

rows.to_csv(path + 'row.csv')
birdstrike_A.to_csv(path + 'birdstrike_A.csv')
birdstrike_B.to_csv(path + 'birdstrike_B.csv')
birdstrike_C.to_csv(path + 'birdstrike_C.csv')
birdstrike_D.to_csv(path + 'birdstrike_D.csv')

birdstrike_A.columns = ['id_aeroporto', 'data', 'data_', 'A_effettoSulVolo', 'id_specie_volatile', 'id_gruppo']
birdstrike_B.columns = ['id_aeroporto', 'data', 'data_', 'A_effettoSulVolo', 'id_specie_volatile', 'id_gruppo']
birdstrike_C.columns = ['id_aeroporto', 'data', 'data_', 'A_effettoSulVolo', 'id_specie_volatile', 'id_gruppo']
birdstrike_D.columns = ['id_aeroporto', 'data', 'data_', 'A_effettoSulVolo', 'id_specie_volatile', 'id_gruppo']

birdstrike = pd.concat([birdstrike_D, birdstrike_A, birdstrike_B, birdstrike_C], axis=0)
birdstrike = birdstrike.drop_duplicates(subset=['data'])
birdstrike.to_csv(path + 'birdstrike_total.csv')

mensili.to_csv(path + 'mensili.csv')

