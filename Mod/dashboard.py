# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import xarray as xr
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import statsmodels as sm
import seaborn as sns
sns.set_style('darkgrid')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def DJF(df):
    c = xr.Dataset(df)
    c = c.sel(time=slice('1979-01-31','2020-01-01'))
    c = c.where((c['time.month']==2) | (c['time.month']==1) | (c['time.month']==12)).dropna(dim='time').to_dataframe()
    c = c.drop(['lat','lon'],axis=1)
    c.index = pd.to_datetime(c.index)
    return c

# Saving pandas.DataFrame's results to variables
BE = DJF(pd.read_pickle('/home/github/Climatology/Mod/Files/b_esperanca.pkl'))
FU = DJF(pd.read_pickle('/home/github/Climatology/Mod/Files/furnas.pkl'))
GB = DJF(pd.read_pickle('/home/github/Climatology/Mod/Files/g_b_munhoz.pkl'))
PR = DJF(pd.read_pickle('/home/github/Climatology/Mod/Files/passo_real.pkl'))
SA = DJF(pd.read_pickle('/home/github/Climatology/Mod/Files/santo_antonio.pkl'))
SS = DJF(pd.read_pickle('/home/github/Climatology/Mod/Files/sao_simao.pkl'))
SO = DJF(pd.read_pickle('/home/github/Climatology/Mod/Files/sobradinho.pkl'))
TU = DJF(pd.read_pickle('/home/github/Climatology/Mod/Files/tucurui.pkl'))


#   Third section             -------------------------------

app.layout = html.Div(children=[
    html.H1(children='Precipitação - Usinas', style={
        'textAlign': 'center'}),

    html.Div(children='''
       Usinas:
    '''),
    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'Furnas', 'value': "FU"},
            {'label': 'Passo Real - Jacuí', 'value': "PR"},
            {'label': 'Foz do Areia', 'value': "FA"},
            {'label': 'Boa Esperanca - Parnaíba', 'value': "BE"},
            {'label': 'Santo Antonio - Madeira', 'value': "SA"},
            {'label': 'São Simão', 'value': "SS"},
            {'label': 'Sobradinho', 'value': "SO"},
            {'label': 'Tucuruí - Tocantins', 'value': "TU"}

        ],
        value='FU'
    ),
    html.Div(id='dd-output-container'),
    
    html.Button('Linear', id='btn-nclicks-1', n_clicks=0),

    html.Button('Trendline', id='btn-nclicks-2', n_clicks=0),
    
])

@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    dash.dependencies.Input('demo-dropdown', 'value'),
    dash.dependencies.Input('btn-nclicks-1', 'n_clicks'),
    dash.dependencies.Input('btn-nclicks-2', 'n_clicks'))
def update_output(value, btn1, btn2):
    changed_graph = [p['prop_id'] for p in dash.callback_context.triggered][0]
    var = FU

    if value == "FU":
        var = FU
    elif value == "PR":
        var = PR
    elif value == "GB":
        var = GB
    elif value == "BE":
        var = BE
    elif value == "SA":
        var = SA
    elif value == "SS":
        var = SS
    elif value == "SO":
        var = SO
    elif value == "TU":
        var = TU  
    

    if 'btn-nclicks-1' in changed_graph:
        fig = px.line(var, x=var.index.astype(str), y='precip', title='Precipitação ao longo do shape da usina no período úmido | 01/1979 - 11/2020')
        
    elif 'btn-nclicks-2' in changed_graph:
        fig = px.scatter(var, x=var, y='precip', title='Precipitação ao longo do shape da usina no período úmido | 01/1979 - 11/2020', trendline='ols')
        
    else:
        fig = px.line(var, x=var.index, y='precip', title='Precipitação ao longo do shape da usina no período úmido | 01/1979 - 11/2020')
        
    return dcc.Graph(id='example-graph', figure=fig)
    

if __name__ == '__main__':
    app.run_server(debug=True)