import pandas as pd
import requests
import datetime


def api_niteroi(api_type='chuva'):
    """
    Retorna um DataFrame com os dados da API do Niterói.

    'api_type' pode ser 'chuva' ou 'tempo'.
    'estacao' pode ser 'Barreto 1'
    """

    api = requests.get('http://svidaniteroi.com.br/'+ api_type +'_api')

    with open(api_type + '.csv', 'w') as f:
        f.write(api.text)

    data=pd.read_csv(api_type + '.csv', sep=';') 
    
    if api_type == 'chuva':
        new_column_names = ['Lat','lon', 'estação', 'cidade', 'data', 'chuva 15m', 'chuva 1h', 'chuva 4h', 'chuva 24h', 'chuva 96h', 'chuva 30d', 'atual ou atrasado', 'fonte do dado']
        pos = 0
        insert_dict = dict()
        for i in data.columns:
            insert_dict.update({i:i})
            pos+=1 
        
        updated_data = data.append(insert_dict, ignore_index=True)
        updated_data.columns=new_column_names

        return updated_data

    elif api_type == 'tempo':
        new_column_names=['Nome', 'lat', 'lon', 'temperatura', 'umidade', 'direção do vento', 'velocidade do vento', 'hora do dado', 'sensação termica']
        pos = 0
        insert_dict = dict()
        for i in data.columns:
            insert_dict.update({i:i})
            pos+=1 
                
        print(insert_dict)
        updated_data = data.append(insert_dict, ignore_index=True)
        updated_data.columns=new_column_names

        return updated_data

def getting_today_precipitation(estacao):
    """
    Retorna um valor com a chuva do dia de hoje.
    """

    data = api_niteroi('chuva')

    filtering_station = data[data['estação']==estacao]
    precip_24_value = filtering_station['chuva 24h'].values[0]
    
    return precip_24_value


hour = datetime.datetime.now().hour
hourly_barreto = api_niteroi('chuva')
hourly_barreto.to_csv(str(hour) + '_rain.csv')