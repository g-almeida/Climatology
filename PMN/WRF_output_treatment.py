    """
        Script designated to treat the output of the WRF model.
            The Functions in this section are able to select the interval of time. And also formats each of the desired files.
    """

import os
import shutil
import datetime
import pandas as pd

origin_path = "/home/data/PMN/WRF_extrai_rn/"
destination_path = "/home/data/PMN/WRF_Barreto_10-7_20-10/"

files = os.listdir(origin_path)

# Filtering Barreto files.
barreto_list = []
for cada in files:
  if 'Barreto' in cada and 'Zone' not in cada:
    barreto_list.append(cada)


def files_date_filter(start_date, end_date):
    '''
    Filters data for starting and ending dates.
    '''
    b_prec = [] # lista dos arquivos de chuva em barreto
    b_vent = [] # lista dos arquivos de vento em barreto
    others = [] # lista dos restantes
    for x in barreto_list:
      if 'prec' in x:
        b_prec.append(x)
      elif 'vent' in x:
        b_vent.append(x)
      else:
        others.append(x)

    datefilter=[]
    while start_date <= end_date:
      datestr = start_date.strftime("%Y-%m-%d").replace('-','')
      for cada in others:
        if datestr in cada:
          datefilter.append(cada)    
      start_date+=datetime.timedelta(days=1)

    return datefilter
  
date = datetime.datetime(2021,7,10)
date2 = datetime.datetime(2021,10,20)
datefilter = files_date_filter(date, date2)

for file in files:
  if file in datefilter:
    # First path is the WRF provided data
    # Second path is the destination path for the filtered data (station and date)
    shutil.move(origin_path+ file, destination_path)


#  -----------   Files treatment

def adaptingTXT(path):
  data = pd.read_csv(path, sep='\s+')
  pt_cols=['DATA', 'HORA','Barreto','Pto N','Pto S','Pto E','Pto W','Pto SE','Pto NE','Pto SW','Pto NW']
  vent_cols=['DATA',    'HORA',     'Barreto_ws10', 'Barreto_wd10', 'Pto N_ws10',   'Pto N_wd10',   'Pto S_ws10',   'Pto S_wd10',   'Pto E_ws10',   'Pto E_wd10',   'Pto W_ws10',   'Pto W_wd10',   'Pto SE_ws10',  'Pto SE_wd10' , 'Pto NE_ws10' , 'Pto NE_wd10'  ,'Pto SW_ws10'  ,'Pto SW_wd10','Pto NW_ws10', 'Pto NW_wd10']

  try:
    data.drop(['Pto.8', 'SE_ws10', 'Pto.9', 'SE_wd10', 'Pto.10', 'NE_ws10', 'Pto.11',
       'NE_wd10', 'Pto.12', 'SW_ws10', 'Pto.13', 'SW_wd10', 'Pto.14',
       'NW_ws10', 'Pto.15', 'NW_wd10'], axis=1, inplace=True)
    data.columns=vent_cols
  except:
    data.drop(['Pto.4', 'SE', 'Pto.5', 'NE', 'Pto.6', 'SW', 'Pto.7','NW'], axis=1, inplace=True)
    data.columns=pt_cols
  
  return data

# creating the directory inside the repository
os.makedirs('/home/github/Climatology/PMN/files/OctoberWRF')

path = destination_path
for cada in os.listdir(path):
  path2 = path+cada
  print(path2)
  csv = adaptingTXT(path2)
 
  csv.to_csv('/home/github/Climatology/PMN/files/OctoberWRF/'+cada[:-3]+'csv')


# Check the concatening stuff on filtering notebook on colab from glmalmeida@id.uff.br