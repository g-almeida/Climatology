# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import statsmodels as sm

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Saving pandas.DataFrame's results to variables
FU = pd.read_pickle('Furnas.pkl')
PR = pd.read_pickle('Passo Real - Jacui.pkl')
FA = pd.read_pickle('Foz do Areia.pkl')
BE = pd.read_pickle('Boa Esperanca - Parnaiba.pkl')
SA = pd.read_pickle('Santo Antonio - Madeira.pkl')
SS = pd.read_pickle('Sao Simao.pkl')
SO = pd.read_pickle('Sobradinho.pkl')
TU = pd.read_pickle('Tucurui - Tocantins.pkl')


#   Third section             -------------------------------

app.layout = html.Div(children=[
    html.H1(children='Usin precipitation', style={
        'textAlign': 'center'}),

    html.Div(children='''
        Choose Usin:
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
    elif value == "FA":
        var = FA
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
        fig = px.line(var, x='time', y='precip', title='Precipitation along the usin shape | 01/1979 - 11/2020')
        
    elif 'btn-nclicks-2' in changed_graph:
        fig = px.scatter(var, x='time', y='precip', title='Precipitation along the usin shape | 01/1979 - 11/2020', trendline='ols')
        
    else:
        fig = px.line(var, x='time', y='precip', title='Precipitation along the usin shape | 01/1979 - 11/2020')
        
    return dcc.Graph(id='example-graph', figure=fig)
    

if __name__ == '__main__':
    app.run_server(debug=True)