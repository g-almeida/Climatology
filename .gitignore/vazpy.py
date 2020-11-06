import xarray as xr
import numpy as np
import pandas as pd
import os

def normalize(ds):
    '''
    Receives a xarray.Dataset object and returns a normalized one.
    '''
    minimum = []
    maximum = []
    for index in ds.data_vars:
        minimum.append(ds.data_vars[index].values.min())
        maximum.append(ds.data_vars[index].values.max())
        datamin = np.min(minimum)
        datamax = np.max(maximum)
    norm = (ds-datamin)/(datamax-datamin)
    return norm

def slicer(dataset, series='all-time'):
    """
    Receives a dataset to be time-sliced according to series parameter.
    Call signature::

    self.slicer(series)

    Parameters
    ----------
    series: 'all-time', 'rainy season' or list
    Slicing pattern. 'all-time' returns the whole time series, while 'rainy season' returns time series sliced from october to april for each year (as described in brazilian weather). A list of integers should be given if the user wants to manually select a month or a list of months. For example, series = [1, 4, 6] returns the given dataset sliced at January, April and June.

    Returns
    ----------
    obj: climsy.dataset

    """
    ds = dataset
    #if type(ds) == pandas.DataFrame:
    #    ds = ds.to_dataset
    if series == 'all-time':
        data = ds
    elif series == 'rainy season':
        ends = ds.where(ds['time.month']>=10, drop=True)
        begins = ds.where(ds['time.month']<4, drop=True)
        data = xr.concat([ends, begins], dim = 'time')
        data = data.sortby(data.time).dropna(dim='time')
    elif type(series) == list:
        monthly_data = []
        for month in series:
            sliced_data = ds.where(ds['time.month']==month, drop=True)
            monthly_data.append(sliced_data)
        data = xr.concat(monthly_data, dim = 'time')
        data = data.sortby(data.time).dropna(dim='time')
    return data

def dfvazoes(csv_path, series, usinas):
    '''
    Receives series('rainy season' or 'all-time') and usinas (string with the identification number of the plant)
    Returns a normalized pandas.Dataframe object
    '''
    #criando duas listas vazias para receber a vazão por usina e a vazão normalizada
    vaz_usina = []

    #O loop recebe o arquivo de vazões e dá o merge pra juntar o que for do south com o resto do brasil.
    #depois disso, é atrelado a uma lista de vazões as usinas fornecidas pelo usuário, selecionadas
    #pelo código da usina. Depois de juntar all as usinas solicitadas, dá o merge pra criar um dataset
    #apenas destas
    vazao_total = xr.merge(vazoes(csv_path, regiao='all'))

    for usina in usinas:
        vaz_usina.append(normalize(vazao_total[usina].resample(time="MS").mean().to_dataset()))
    vazds = xr.merge(vaz_usina)


    #normalizando os valores das vazões, dado que algumas das usinas podem apresentar vazões muito
    #maiores umas das outras. Isso nivela os valores. Em seguida, é criado um dataframe com todos os valores
    #de vazão normalizados
    #normvaz = normalize(vazds.resample(time="MS").mean())
    #df = normvaz.to_dataframe()

    #df = vazds.to_dataframe()

    #recorte da série. caso series= 'série histórica', o df permanece o mesmo, mas quando
    #series= 'período úmido', é necessário recortar o dataframe apenas para os meses entre
    #Outubro (10) e Março (3). Após o recorte, são calculadas as médias mensais e os valores
    #NA são removidos do dataframe. O mesmo procedimento é feito caso o valor de seriesseja uma
    #lista de meses (int)

    # PODE SER QUE O SLICER NÃO FUNCIONE PRA DATAFRAME AINDA !!!
    df = slicer(vazds, series=series)
    #df = df.to_dataframe()
    df.attrs['analysis'] = "water outlet"
    #df.rename({'6': 'Furnas', '168': 'Sobradinho', '74': 'Foz do Areia'}, axis=1, inplace=True)

    return df

def vazoes(csv_path, regiao = 'all'):
    '''
    Receives the 'csv' path and the region type ('all', 'south')
    Works with "vazoesdiarias.csv"
    Returns a xarray.Dataset object w/ the new specifications
    '''
    vazoes = pd.read_csv(os.path.join(os.getcwd(), "csv", csv_path), sep=";", header=5, index_col=0, decimal = ',', low_memory=False)
    vazoes.index = pd.to_datetime(vazoes.index, errors='coerce')
    vazoes_br = vazoes.loc['1979-01-01' : '2019-01-01', ['246', '34', '237', '240', '33', '24', '6', '18', '156', '168', '275', '287', '285', '190', '254']]
    vazoes_br = vazoes_br.apply(pd.to_numeric, errors='ignore')
    vazoes_south = vazoes.loc['1979-01-01' : '2019-01-01', ['111', '217', '74', '78', '66', '63', '61']]
    vazoes_south = vazoes_south.apply(pd.to_numeric, errors='ignore')
    br = xr.Dataset.from_dataframe(vazoes_br)
    br = br.rename({'index': 'time'})
    br = br.sortby('time')
    south = xr.Dataset.from_dataframe(vazoes_south)
    south = south.rename({'index': 'time'})
    south = south.sortby('time')
    if regiao == 'all':
        return [br, south]
    if regiao == 'south':
        return south
    if regiao == 'br':
        return br


def vazcorrs(atmds, regiao, freq, season=None, mes=None, savefig=False, save_csv=False, csv_type=None):

    '''
    Calcula as correlações entre os índices (bloqueio, vorticidade, divergência) e as vazões naturais das usinas.

    Os valores das vazões são normalizados para que, em razão da diferença entre as dimensões espaciais das usinas, não haja discrepância entre os dados.

    Call signature.:

        vazcorrs(atmds, regiao, freq, season=None, mes=None, savefig=False, save_csv=False, csv_type=None)

    Parameters
    ----------

    atmds: list of <xarray.Dataset>

        Lista de datasets de vorticidade e divergência.

        Função recebe até *três* datasets, sendo que o último da lista (atmds[2]), nesse estudo, deve ser o *dataset da vorticidade em 500 hpa*.

        EX.: atmds = [div850, vort850, vort500] --> vort500 por último (!!) --> div850 e vort850 não importa a ordem

    regiao: str, {"sul", "todas", "br"}

        Região das usinas em análise.

            Se "sul", serão computadas somente usinas do sul do Brasil.

            Se "br", todas as usinas *exceto* as do sul.

            Se "todas", todas usinas serão computadas.

    freq: str, {"M" (ou 'month'), "season"}

        Determina o tipo de análise da série de dados. "Season" corresponde a análise sazonal e "M" (ou "month") mensal.

    season: str, {"DJF", "MAM", "JJA", "SON"}

        Estação do ano em análise. Somente se * freq == "season" *

    mes: int, range(1, 13)

        Mês do ano em análise. Somente se * freq == "M" *

    Kwargs
    ------

    savefig: bool

        Salva a imagem gerada no diretório segundo o mesmo caminho em que 'arquivo.ipynb' está salvo e sendo executado.

        O nome da imagem é pré-definido em função do tipo de bloqueio (Completo, Sul ou Norte), região (*regiao*), frequência (*freq*) e *season*/*mes* (depende do valor fornecido para *freq*)

        Ex.: 'corr_VAZ_BloqueioNorte_SUL_DEC_1979_2010.png' --> nome da imagem equivalente ao Bloqueio Norte, usinas do sul, mês de dezembro e periodo de 1979 a 2010.

    save_csv: bool

        Salva o arquivo 'csv' do dataframe correspondente ao valor fornecido em *csv_type*.

        Assim como *savefig*, o nome do 'csv' é pré-definido em função do tipo de bloqueio (Completo, Sul ou Norte), região (*regiao*), frequência (*freq*) e *season*/*mes* (depende do valor fornecido para *freq*)

        Ex.: 'corr_BloqueioNorte_SUL_DEC_1979_2010.csv' --> nome do 'csv' equivalente ao dataframe de correlações para Bloqueio Norte, usinas do sul, mês de dezembro e periodo de 1979 a 2010.


    csv_type: str, {"corr", "ind"}

        Determina o dataframe a ser exportado em formato 'csv'.

            Se * csv_type == "corr" *, o 'arquivo.csv' exportado é aquele correspondente ao dataframe das correlações.

            Se * csv_type == "ind" *, o 'arquivo.csv' exportado é aquele correspondente ao dataframe de índices antes de serem feitas as correlações.

        Ex.: 'corr_VAZ_BloqueioNorte_SUL_DEC_1979_2010.csv' --> dataframe de correlações (*csv_type == 'corr'*)

        Ex.: 'df_VAZ_BloqueioNorte_SUL_DEC_1979_2010.csv' --> dataframe do índices sem correlações (*csv_type == 'ind'*)

    Returns
    -------

    Figure: <class 'matplotlib.figure.Figure'>

        Figura do gráfico de correlações entre os índices e as vazões naturais das usinas.

    '''

    for ds in atmds:
        indice_bloq = ds
        print(ds)
        ext_name = ds.dataset.to_array().isel(variable=0)
        var = str(ext_name['variable'].values)
        ds = ds.rename({var: var + "_" + ds.level})
        print(ds)

        # DATASETS DOS ÍNDICES (DEPOIS SERÃO CONVERTIDOS EM DATASET)
    indice_bloq = blockix([atmds[0], atmds[1]], bloq='total', freq=freq)

    indice_div_vort = posix([atmds[0], atmds[1]], bloq='total', freq=freq)
    indice_div_vort = indice_div_vort.rename({"vort":"vort_850", "div":"div_850"})

    indice_vort_500 = posix(atmds[2], bloq='total', freq=freq)
    indice_vort_500 = indice_vort_500.rename({"vort":"vort_500"})

    indice_vort_700 = posix(atmds[3], bloq='total', freq=freq)
    indice_vort_700 = indice_vort_700.rename({"vort":"vort_700"})

    indice_div_700 = posix(atmds[4], bloq='total', freq=freq)
    indice_div_700 = indice_div_700.rename({"div":"div_700"})

    # NOMEANDO O BLOQUEIO EM FUNÇÃO DE SUAS LATITUDES (BLOQUEIO NORTE, SUL, COMPLETO)
    # 'BLOQ_NAME' PARA NOME DA IMAGEM SALVA E 'BLOQ_NAME_TITLE' PARA TITULO DA FIGURA
    if indice_div_vort.attrs['lat'][0:5] + '_' + indice_div_vort.attrs['lat'][8:13] == '-10.0_-25.0':
        bloq_name = "BloqueioCompleto"
        bloq_name_title = "Área total"
    if indice_div_vort.attrs['lat'][0:5] + '_' + indice_div_vort.attrs['lat'][8:13] == '-10.0_-17.5':
        bloq_name = "BloqueioNorte"
        bloq_name_title = "Setor norte"
    if indice_div_vort.attrs['lat'][0:5] + '_' + indice_div_vort.attrs['lat'][8:13] == '-17.5_-25.0':
        bloq_name = "BloqueioSul"
        bloq_name_title = "Setor sul"
    print("BLOCK NOME")
    # FORMATANDO O DATASET DE VAZÕES PARA MESMO INTERVALO TEMPORAL DO DATASET DE INDICES_BLOQ / INDICES_DIV_VORT
    if regiao == "todas":
        vaz_br = vazoes(regiao=regiao)[0].sel(time=slice(pd.to_datetime(indice_bloq.time.values[0]), pd.to_datetime(indice_bloq.time.values[-1]))).to_array()
        vaz_sul = vazoes(regiao=regiao)[1].sel(time=slice(pd.to_datetime(indice_bloq.time.values[0]), pd.to_datetime(indice_bloq.time.values[-1]))).to_array()
        vaz_total = [vaz_br, vaz_sul]
        vaz = xr.concat(vaz_total, dim="variable").to_dataset(dim="variable")
    elif regiao == "sul":
        vaz = vazoes(regiao=regiao).sel(time=slice(pd.to_datetime(indice_bloq.time.values[0]), pd.to_datetime(indice_bloq.time.values[-1])))

    # FORMATANDO O DATASET DE VAZÕES PARA SER COERENTE COM O TIPO DE ANÁLISE DO DATASET DE ÍNDICES (DEPENDE DO VALOR DE 'FREQ')
    if freq == 'season':
        normvaz = normalize(vaz.resample(time='QS-DEC').mean()).to_dataframe()
    elif freq == 'M' or freq == 'month':
        normvaz = normalize(vaz.resample(time='MS').mean()).to_dataframe()

    # JUNTANDO OS 4 DATASETS E FORMATANDO EM DATAFRAME (TIRAR A CORRELAÇÃO E MELHOR VISUALIZAÇÃO)
    df = normvaz.join([indice_bloq.to_dataframe(), indice_div_vort.to_dataframe(), indice_vort_500.to_dataframe(), indice_vort_700.to_dataframe(), indice_div_700.to_dataframe()])
    df.fillna(0, inplace=True)
    print(df)
    # REORGANIZANDO AS COLUNAS DO DATAFRAME
    new = []
    col_list = list(df)
    for i in range(-4, 0):
        index = i
        new.append(col_list[index])
    for i in range(0, len(col_list)-4):
        new.append(col_list[i])

    df = df[new]
    df = df.rename(columns={"índice de bloqueios": "Dias de bloqueios", "div_850": "Persistência de divergência (850 hPa)", "vort_850": "Persistência de vorticidade (850 hPa)", "vort_500": "Persistência de vorticidade (500 hPa)", "246":"Porto Primavera", "34":"Ilha Solteira", "237":"Barra Bonita", "240":"Promissão", "33":"São Simão", "24":"Emborcação", "6":"Furnas", "18":"Água Vermelha", "156":"Três Marias", "168":"Sobradinho", "275":"Tucuruí", "287":"Santo Antônio", "285":"Jirau", "190":"Boa Esperança", "254":"Pedra do Cavalo", "111":"Passo Real", "217":"Machadinho", "74":"Foz do Areia", "78":"Salto Osório", "66":"Itaipu", "63":"Rosana", "61":"Capivara"})
    df = df.rename(columns={"vort_700": "Persistência de vorticidade (700 hPa)"})
    df = df.rename(columns={"div_700": "Persistência de divergência (700 hPa)"})
    print(df)
    # ADAPTANDO A ANÁLISE DA SÉRIE DE DADOS COM O VALOR FORNECIDO DE 'FREQ'
    if freq == 'season':
        if season == "DJF":
            df = df[df.index.month==12]
        if season == "MAM":
            df = df[df.index.month==3]
        if season == "JJA":
            df = df[df.index.month==6]
        if season == "SON":
            df = df[df.index.month==9]
    elif freq == 'month' or freq == 'M':
        df = df[df.index.month==mes]

    # DICIONÁRIOS QUE SERÃO CONVERTIDOS PARA DATAFRAME APÓS CÁLCULO DE CORRELAÇÕES
    ixcorrs = {}
    v850corrs = {}
    d850corrs = {}
    v500corrs = {}
    v700corrs = {}
    d700corrs = {}

    # CÁLCULO DAS CORRELAÇÕES COM O TESTE DE SIGNIFÂNCIA (95 %)
    for column in df.columns:
        if stats.pearsonr(df[column], df['Dias de bloqueios'])[1] < 0.05:
            ixcorrs.update({column: stats.pearsonr(df[column], df['Dias de bloqueios'])[0]})
        else:
            ixcorrs.update({column: stats.pearsonr(df[column], df['Dias de bloqueios'])[0]})
        if stats.pearsonr(df[column], df['Persistência de vorticidade (850 hPa)'])[1] < 0.05:
            v850corrs.update({column: stats.pearsonr(df[column], df['Persistência de vorticidade (850 hPa)'])[0]})
        else:
            v850corrs.update({column: stats.pearsonr(df[column], df['Persistência de vorticidade (850 hPa)'])[0]})
        if stats.pearsonr(df[column], df['Persistência de divergência (850 hPa)'])[1] < 0.05:
            d850corrs.update({column: stats.pearsonr(df[column], df['Persistência de divergência (850 hPa)'])[0]})
        else:
            d850corrs.update({column: stats.pearsonr(df[column], df['Persistência de divergência (850 hPa)'])[0]})
        if stats.pearsonr(df[column], df['Persistência de vorticidade (500 hPa)'])[1] < 0.05:
            v500corrs.update({column: stats.pearsonr(df[column], df['Persistência de vorticidade (500 hPa)'])[0]})
        else:
            v500corrs.update({column: stats.pearsonr(df[column], df['Persistência de vorticidade (500 hPa)'])[0]})
        if stats.pearsonr(df[column], df['Persistência de vorticidade (700 hPa)'])[1] < 0.05:
            v700corrs.update({column: stats.pearsonr(df[column], df['Persistência de vorticidade (700 hPa)'])[0]})
        else:
            v700corrs.update({column: stats.pearsonr(df[column], df['Persistência de vorticidade (700 hPa)'])[0]})
        if stats.pearsonr(df[column], df['Persistência de divergência (700 hPa)'])[1] < 0.05:
            d700corrs.update({column: stats.pearsonr(df[column], df['Persistência de divergência (700 hPa)'])[0]})
        else:
            d700corrs.update({column: stats.pearsonr(df[column], df['Persistência de divergência (700 hPa)'])[0]})


    # CONVERTENDO PARA DATAFRAME AS CORRELAÇÕES CALCULADAS COM O TESTE DE SIGNIFICÂNCIA
    df_1 = pd.DataFrame.from_dict(ixcorrs, orient='index', columns=['Dias de bloqueios'])
    df_2 = pd.DataFrame.from_dict(v850corrs, orient='index', columns=['Persistência de vorticidade (850 hPa)'])
    df_3 = pd.DataFrame.from_dict(d850corrs, orient='index', columns = ['Persistência de divergência (850 hPa)'])
    df_4 = pd.DataFrame.from_dict(v500corrs, orient='index', columns = ['Persistência de vorticidade (500 hPa)'])
    df_5 = pd.DataFrame.from_dict(v700corrs, orient='index', columns = ['Persistência de vorticidade (700 hPa)'])
    df_6 = pd.DataFrame.from_dict(d700corrs, orient='index', columns = ['Persistência de divergência (700 hPa)'])

    # GERANDO E FORMATANDO O DATAFRAME FINAL DE CORRELAÇÕES
    df = pd.concat([df_1, df_3, df_2, df_4, df_5, df_6], axis=1)
    corre = df.T.drop(['Dias de bloqueios', 'Persistência de vorticidade (850 hPa)', 'Persistência de divergência (850 hPa)', 'Persistência de vorticidade (500 hPa)', 'Persistência de vorticidade (700 hPa)', 'Persistência de divergência (700 hPa)'], axis=1)

    # PLOTANDO O HEATMAP E ATRIBUINDO AS FORMATAÇÕES E 'DESIGN' NECESSÁRIOS PARA CADA REGIÃO
    if regiao == 'sul':
        corr, ax = plt.subplots(figsize=(12,5))
        corr = sns.heatmap(corre, annot=True, annot_kws={'fontsize':12, 'weight': 'bold'}, cbar_kws={'pad': 0.02}, cmap='RdBu', vmin=-1, vmax=1, center = 0, linewidths=0.5)
        plt.yticks(rotation=0, fontsize=12)
        plt.xticks(rotation=45, fontsize=12, ha="right")

    elif regiao == 'todas':
        corr, ax = plt.subplots(figsize=(47,10))
        corr = sns.heatmap(corre, annot=True, annot_kws={'fontsize':25, 'weight': 'bold'}, cbar=False, cmap="RdBu", linewidths=0.5, vmin=-1, vmax=1, center = 0)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-1.,vmax=1.), cmap="RdBu"), pad=0.02, orientation="vertical", ax=ax)
        cbar.ax.tick_params(labelsize=25)
        plt.xticks(rotation=45, fontsize=25, ha="right")
        plt.yticks(rotation=0, fontsize=25)

    # ESPECIFICAÇÕES DE TÍTULO DA FIGURA
    # PRÉ-DEFININDO OS NOMES DOS CSVs E IMAGENS A SEREM EXPORTADOS
    if freq == 'season':
        if regiao == 'sul':
            ax.set_title(bloq_name_title + ' x Usinas (' + regiao.upper() + ') | ' + str(pd.to_datetime(atmds[0].dataset.time.values[0]))[:4] + ' - ' + str(pd.to_datetime(atmds[0].dataset.time.values[-1]))[:4] + ' | ' + season + "\n\n\u03B1 = 0.05\n", fontsize=15, style="oblique")
        else:
            ax.set_title(bloq_name_title + ' x Usinas (' + regiao.upper() + ') | ' + str(pd.to_datetime(atmds[0].dataset.time.values[0]))[:4] + ' - ' + str(pd.to_datetime(atmds[0].dataset.time.values[-1]))[:4] + ' | ' + season + "\n\n\u03B1 = 0.05\n", fontsize=37, style="oblique")
        if savefig == True:
            plt.savefig('corr_VAZ_' + bloq_name + '_' + regiao.upper() + '_' + season + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[0]))[:4] + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[-1]))[:4] + '.png', format='png', dpi=200, bbox_inches="tight")
        if save_csv == True:
            if csv_type == "corr":
                corre.to_csv('corr_VAZ_' + bloq_name + '_' + regiao.upper() + '_' + season + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[0]))[:4] + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[-1]))[:4] + '.csv')
            elif csv_type == "ind":
                df.to_csv('df_VAZ_' + bloq_name + '_' + regiao.upper() + '_' + season + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[0]))[:4] + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[-1]))[:4] + '.csv')

    elif freq == 'month' or freq == 'M':
        if regiao == "sul":
            ax.set_title('Dias de bloqueio x Usinas | \u03B1 = 0.05 | ' + bloq_name_title + ' | ' + str(pd.to_datetime(atmds[0].dataset.time.values[0]))[:4] + ' - ' + str(pd.to_datetime(atmds[0].dataset.time.values[-1]))[:4] + ' | ' + calendar.month_abbr[mes].upper() + '\n', fontsize=15, style="oblique", pad=0.1)
        else:
            ax.set_title('Dias de bloqueio x Usinas | \u03B1 = 0.05 | ' + bloq_name_title + ' | ' + str(pd.to_datetime(atmds[0].dataset.time.values[0]))[:4] + ' - ' + str(pd.to_datetime(atmds[0].dataset.time.values[-1]))[:4] + ' | ' + calendar.month_abbr[mes].upper() + '\n', fontsize=37, style="oblique")

        if savefig == True:
            plt.savefig('corr_VAZ_' + bloq_name + '_' + regiao.upper() + '_' + calendar.month_abbr[mes].upper() + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[0]))[:4] + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[-1]))[:4] + '.png', format='png', dpi=200, bbox_inches="tight", transparent=True)
        if save_csv == True:
            if csv_type == "corr":
                corre.to_csv('corr_VAZ_' + bloq_name + '_' + regiao.upper() + '_' + calendar.month_abbr[mes].upper() + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[0]))[:4] + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[-1]))[:4] + '.csv')
            elif csv_type == "ind":
                df.to_csv('df_VAZ_' + bloq_name + '_' + regiao.upper() + '_' + calendar.month_abbr[mes].upper() + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[0]))[:4] + '_' + str(pd.to_datetime(atmds[0].dataset.time.values[-1]))[:4] + '.csv')
