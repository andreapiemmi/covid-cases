#install all packages
from dash import Dash, html, dcc, Input, Output, callback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.request import urlopen
import json
import dash_daq as daq
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

import statsmodels.api as sm

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://raw.githubusercontent.com/andreapiemmi/covid-cases/main/Confirmed_Cases_Dataset.csv')
df=df.rename(columns={'Unnamed: 0':'Date'}).set_index('Date')
df.index, numdate = pd.DatetimeIndex(df.index), [x for x in range(len(df.index))]  


def rolling_window_estimation(log_series,window_size,dailyfreq_dataset):
  rolling_dates=[]
  rolling_doubling_times=[]

  for i in log_series.index[window_size:-1]:
    end_date=i
    start_date=log_series.index[int(np.where(log_series.index==i)[0])-window_size]
    window_data=dailyfreq_dataset.loc[ start_date : end_date ]
    X, y = sm.add_constant(np.arange(1,window_data.shape[0]+1)), window_data
    model=sm.OLS(y,X).fit()
    incercept, slope = model.params


    doubling_time = np.log(2) / ((1+slope)**7-1) #Convert the estimated growth rate into weekly measure
   
    
    rolling_dates.append(i)
    rolling_doubling_times.append(doubling_time)



  return rolling_dates,rolling_doubling_times#,ci_lower,ci_upper


df_w=pd.read_csv('https://raw.githubusercontent.com/andreapiemmi/covid-cases/main/Confirmed_Cases_Dataset.csv').rename(
    columns={'Unnamed: 0':'Date'}).set_index('Date')
df_w.index=pd.DatetimeIndex(df.index)
df_w_log=np.log(1+df_w)

df_daily=pd.read_csv('https://raw.githubusercontent.com/andreapiemmi/covid-cases/main/Confirmed_Cases_Daily.csv').rename(
    columns={'Unnamed: 0':'Date'}).set_index('Date')
df_daily.index=pd.DatetimeIndex(df_daily.index)
df_daily['Belarus'][np.logical_or(df_daily['Belarus']==df_daily.iloc[-1,:]['Belarus'],
                              df_daily['Belarus']==df_daily[df_daily['Belarus']!=df_daily.iloc[-1,:]['Belarus']].iloc[-1]['Belarus'])]=None

df_daily_log=np.log(1+df_daily)

WS=12
Doubling_Time={}
for c in df_w.columns:
    Doubling_Time[c]=rolling_window_estimation(log_series=df_w_log[c],window_size=WS,dailyfreq_dataset=df_daily_log[c])[1]

Doubling_Time=pd.DataFrame(Doubling_Time).set_index(df_w_log.index[WS:-1])

numdate1=[x for x in range(len(Doubling_Time.index))]  

app.layout = html.Div(children=[
    
    html.Div([
        html.Div([html.H1('Covid19 Cases Evolution')],
                 style={'width':'45%','display':'inline-block'}),
        html.Div([html.P('Total Cases')],
                 style={'width':'10%','display':'inline-block',
                        'margin': '0px 0px 0px 0px','font-weight':'bold',
                        'font-size':'25px'}),
        html.Div([daq.ToggleSwitch(id='Total-or-New',value=False,size=80)],
                 style={'width':'35%','display':'inline-block'}),
        html.Div([html.P('New Cases')],
                 style={'width':'10%','display':'inline-block',
                        'margin': '0px 0px 0px 0px','font-weight':'bold',
                        'font-size':'25px'})
        ],className='row'),

     html.Div(children=[
        # 1st column of 2nd row
        html.Div([
            dcc.Graph(
            id='EuropeMap'
        ),
            dcc.Slider(id='Time-Slider',min=numdate[0],max=numdate[-1],value=numdate[-1],
        marks={numdate[0]:df.index[0].strftime('%d%b%y'),numdate[-1]:df.index[-1].strftime('%d%b%y')}),
            dcc.Graph(id='Cumulative_Series')
        ], style={'width':'60%', 'display': 'inline-block'}),

        # 2nd column of 2st row
        html.Div([
            html.Div(id='Selected_Date',style={'font-size': '50px'}),
            html.Hr(style={'margin':'0px 0px 0px 30px'}),
            html.Div(id='Country_in_Selection',
                     style={'text-align':'center','vertical-align':'top','font-size':'26px',
                            'margin': '0px 5px 5px 20px'}),
            dcc.Graph(id='Absolute-Growth'),
            dcc.Graph(id='Log-Growth')
        ], style={'width':'40%','display': 'inline-block','text-align': 'left','vertical-align': 'top'}),

    ], className='row'),
     
     html.Br(),html.Br(),
              html.Div([html.Div([html.H1('Doubling Time of Covid19 Cases')],
                  style={'width':'45%','display':'inline-block'})],className='row'),
     html.Div(children=[        
         html.Div([
             dcc.Graph(id='Doubling-Time-Fig')]),
         html.Div([
             dcc.Slider(id='Time-Slider2',min=numdate1[0],max=numdate1[-1],value=numdate1[-1],
                        marks={numdate1[0]:Doubling_Time.index[0].strftime('%d%b%y'),numdate1[-1]:Doubling_Time.index[-1].strftime('%d%b%y')})
             ])
         ],className='row'),
     
     html.Div(children=[
         html.Div([
             dcc.Graph(id='EuropeMap2')
             ],style={'width':'60%','display':'inline-block'}),
         html.Div([
             html.Div(id='Date2',style={'font-size': '35px','display':'inline-block','text-aling':'center','vertical-align':'top'}),
             html.Div(id='Selected-Country2',style={'font-size': '35px','display':'inline-block','text-aling':'center','vertical-align':'top'}),
             html.Hr(style={'margin':'0px 0px 0px 30px'}),
             dcc.Graph(id='Doubling-Time_v_1stDiff-in-Log')
             ],style={'width':'40%','display':'inline-block','vertical-align':'top','text-align':'center'})

         ],className='row'),
     
     html.Br(),html.Br(),
              html.Div([html.Div([html.H1('Exponential Growth in Covid19 cases')],
                  style={'width':'65%','display':'inline-block'})],className='row'),
              
    html.Div(children=[
        html.Div([
        html.Hr(style={'margin':'0px 0px 0px 30px'}),
        dcc.Graph(id='Exponential-Fit')
        ],style={'width':'50%','display':'inline-block','vertical-align':'top','text-align':'center'}),
        html.Div([
            dcc.Graph(id='EuropeMap3'),
            dcc.RangeSlider(id='Time-Slider3',min=numdate[0],max=numdate[-1],value=[numdate[0],numdate[-1]],
                       marks={numdate[0]:df.index[0].strftime('%d%b%y'),numdate[-1]:df.index[-1].strftime('%d%b%y')})
        ],style={'width':'50%','display':'inline-block'})
    ],className='row'),
    
    html.Div(children=[
        html.Div([
            html.Div(id='Selected-Country3',
                     style={'font-size': '25px','display':'inline-block','text-aling':'left','vertical-align':'top'}),
            html.Div(id='Selected-Date3',
                     style={'font-size': '25px','display':'inline-block','text-aling':'left','vertical-align':'top'}),
            html.Div(dcc.Graph(id='provvi1'))
            ],style={'width':'80%','display':'inline-block','vertical-align':'top','text-align':'left'})
        #Put another columns here
        
        ],className='row')
    

])


@callback(
    Output('EuropeMap','figure'),
    Output('Selected_Date', 'children'),
    Output('Cumulative_Series','figure'),
    Input('Time-Slider', 'value'),
    Input('Total-or-New','value'))
def update_map(time,Tot_or_New):
  if Tot_or_New==False:
      dff = df.loc[df.index[time],:]   
  else:
      dff = df.diff().loc[df.index[time],:]
  
  with urlopen('https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson') as response:
    countries = json.load(response)
  
  continental_extremes={'N':(81.843056, 59.239444),
                      'S':(34.833333, 24.083333),
                      'W':(39.495222, -31.275),
                      'E':(75.5, 60)}

  dff=dff.rename(index={'Czechia':'Czech Republic','Moldova':'Republic of Moldova',
                        'North Macedonia':'The former Yugoslav Republic of Macedonia'})
  fig = px.choropleth_mapbox(dff, geojson=countries, 
                           locations=[*dff.index],
                           featureidkey ='properties.NAME',color=dff.values,
                           color_continuous_scale='darkmint',
                           mapbox_style = 'open-street-map',
                           zoom=2.4, center = {"lat": (continental_extremes['N'][0]+continental_extremes['S'][0])/2 -1.5,
                                             "lon": (continental_extremes['W'][1]+continental_extremes['E'][1])/2 },
                           opacity=0.5
                          )
  fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
  
  if Tot_or_New==False:
      fig1=px.line(df.iloc[:time,])
      fig1.layout.yaxis.title="Number of Confirmed Cases"
  else:
      fig1=px.line(df.diff().iloc[:time,])
      fig1.layout.yaxis.title="Number of New Cases"

  fig1.update_layout(height=500, width=850,margin={"r":0,"t":0,"l":0,"b":0},template='plotly_white',
                                              legend=dict(orientation="h",yanchor="top",y=-0.2,xanchor="left",x=0) ) 
  
    
#.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.4))  
  return fig, df.index[time].strftime('%d %B %Y'),fig1

@callback(
    Output('Country_in_Selection','children'),
    Output('Absolute-Growth','figure'),
    Output('Log-Growth','figure'),
    Input('EuropeMap','clickData'),
    Input('Total-or-New','value')
    )
def country_selection(s,Tot_or_New):
    dff=df.rename(columns={'Czechia':'Czech Republic',
                           'Moldova':'Republic of Moldova',
                           'North Macedonia':'The former Yugoslav Republic of Macedonia'})
    if s is None:
        location='Switzerland'
    else:
        location=s['points'][0]['location']
        
    fig=go.Figure(px.line(np.log(1+dff.drop(columns=[location])))).update_traces(opacity=.3,showlegend=False)

    fig.add_trace(px.line(np.log(1+dff[[location]])).update_traces(
            line_color='teal',line_width=4,showlegend=False,opacity=1).data[0])
    
    fig.update_layout(height=500,margin={"r":0,"t":0,"l":0,"b":0},template='plotly_white')
    fig.layout.yaxis.title="log(1+Cases)"
    
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    if Tot_or_New==True:
        fig1_p1 = px.line(dff.diff()[location], render_mode="webgl",).update_traces(line_color='teal')
        fig1_p2 = px.line(np.log(1+dff).diff()[location], render_mode="webgl",).update_traces(yaxis="y2",line_color='slategray')

    else:
        fig1_p1=px.line(dff[location], render_mode="webgl",).update_traces(line_color='teal')
        fig1_p2=px.line(np.log(1+dff[location]), render_mode="webgl",).update_traces(yaxis="y2",line_color='slategray')
     
    fig1.add_traces(fig1_p1.data + fig1_p2.data).update_layout(margin={"r":0,"t":0,"l":0,"b":0},showlegend=False,
                       template='plotly_white')
    fig1.layout.yaxis.title="Cases"
    fig1.layout.yaxis.color="teal"
    fig1.layout.yaxis2.type="log"
    fig1.layout.yaxis2.title="log(1+Cases)"
    fig1.layout.yaxis2.color = 'darkgray'

    
    fig1
        
    return location,fig,fig1


@callback(
    Output('Doubling-Time-Fig','figure'),
    Output('EuropeMap2','figure'),
    Output('Date2','children'),
    Input('Time-Slider2','value')
    )
def doubling_time_func(t):
    dff=pd.DataFrame(Doubling_Time).iloc[:t,]
    continental_extremes={'N':(81.843056, 59.239444),
                      'S':(34.833333, 24.083333),
                      'W':(39.495222, -31.275),
                      'E':(75.5, 60)}

    with urlopen('https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson') as response:
        countries = json.load(response)
  

    dff=dff.rename(columns={'Czechia':'Czech Republic','Moldova':'Republic of Moldova',
                        'North Macedonia':'The former Yugoslav Republic of Macedonia'})
    figMap = px.choropleth_mapbox(dff, geojson=countries, 
                           locations=[*dff.columns],
                           featureidkey ='properties.NAME',color=dff.iloc[-1,:].values,
                           color_continuous_scale='ice',
                           mapbox_style = 'open-street-map',
                           zoom=2.4, center = {"lat": (continental_extremes['N'][0]+continental_extremes['S'][0])/2 -1.5,
                                             "lon": (continental_extremes['W'][1]+continental_extremes['E'][1])/2 },
                           opacity=0.5,
                          )
    figMap.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig=px.line(dff).update_layout(height=300,margin={"r":0,"t":0,"l":0,"b":0})
    fig.layout.yaxis.title="Doubling Time (in weeks)"
    
    return fig, figMap, Doubling_Time.index[t].strftime('%d %B %Y')


@callback(
    Output('Doubling-Time_v_1stDiff-in-Log','figure'),
    Output('Selected-Country2','children'),
    Input('EuropeMap2','clickData')
    )
def Comparison_DT_1stDiff_Log(s):
    dff=df.rename(columns={'Czechia':'Czech Republic',
                           'Moldova':'Republic of Moldova',
                           'North Macedonia':'The former Yugoslav Republic of Macedonia'})
    doubt=Doubling_Time.rename(columns={'Czechia':'Czech Republic',
                           'Moldova':'Republic of Moldova',
                           'North Macedonia':'The former Yugoslav Republic of Macedonia'})
    if s is None:
        location='Switzerland'
    else:
        location=s['points'][0]['location']
        
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig_p1= px.line(doubt[location], render_mode="webgl",).update_traces(line_color='teal')
    fig_p2 = px.line(np.log(1+dff).diff()[location], render_mode="webgl",).update_traces(yaxis="y2",line_color='darkgray')
    fig.add_traces(fig_p1.data + fig_p2.data).update_layout(margin={"r":0,"t":0,"l":0,"b":0},showlegend=False,
                       template='plotly_white')
    fig.layout.yaxis2.type="log"
    fig.layout.yaxis.title="Doubling Time (in weeks)"
    fig.layout.yaxis.color = 'teal'
    fig.layout.yaxis2.title="1stDiff in log(1+Cases)"
    fig.layout.yaxis2.color = 'darkgray'
    
    return fig, ' , '+location

@callback(
    Output('EuropeMap3','figure'),
    Input('Time-Slider3','value'),
    )
def Populate_Map3(t):
    t=t[1]
    dff = df.loc[df.index[t],:]
    with urlopen('https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson') as response:
        countries = json.load(response)
  
    continental_extremes={'N':(81.843056, 59.239444),
                      'S':(34.833333, 24.083333),
                      'W':(39.495222, -31.275),
                      'E':(75.5, 60)}

    dff=dff.rename(index={'Czechia':'Czech Republic','Moldova':'Republic of Moldova',
                        'North Macedonia':'The former Yugoslav Republic of Macedonia'}).drop(
                            columns=['Belarus'])
    fig = px.choropleth_mapbox(dff, geojson=countries, 
                           locations=[*dff.index],
                           featureidkey ='properties.NAME',color=dff.values,
                           color_continuous_scale='tempo',
                           mapbox_style = 'open-street-map',
                           zoom=2.4, center = {"lat": (continental_extremes['N'][0]+continental_extremes['S'][0])/2 -1.5,
                                             "lon": (continental_extremes['W'][1]+continental_extremes['E'][1])/2 },
                           opacity=0.5,
                          )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@callback(
    Output('Exponential-Fit','figure'),
    Output('Selected-Date3','children'),
    Output('Selected-Country3','children'),
    Input('Time-Slider3','value'),
    Input('EuropeMap3','clickData')
    )
def Exp_Smoo(t,co_selected):

    dff=df.rename(columns={'Czechia':'Czech Republic',
                           'Moldova':'Republic of Moldova',
                           'North Macedonia':'The former Yugoslav Republic of Macedonia'}).drop(
                               columns=['Belarus']).iloc[t[0]:t[-1],]
    if co_selected is None:
        location='Switzerland'
    else:
        location=co_selected['points'][0]['location']
        
    hlt_model1 = Holt(dff[location]+1,exponential=True).fit()
    px.line(pd.concat([dff[location]+1,hlt_model1.fittedvalues],axis=1))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig_p1= px.line(pd.concat([dff[location]+1,hlt_model1.fittedvalues],axis=1))
    fig_p2 = px.line( dff[location]+1-hlt_model1.fittedvalues ).add_hline(y=0).update_traces(yaxis="y2",line_color='slategray')
    fig.add_traces(fig_p1.data + fig_p2.data).update_layout(margin={"r":0,"t":0,"l":0,"b":0},showlegend=False,
                        template='plotly_white')
    fig.layout.yaxis.title="Confirmed Cases"
    fig.layout.yaxis2.title="Residual"
    
    accuracy=1-np.sum((dff[location]+1-hlt_model1.fittedvalues)**2)/np.sum((dff[location]-dff[location].mean())**2)
    accuracy=100*round(accuracy,4)
    return fig, ', '+dff.index[0].strftime('%d %b %y')+'-'+dff.index[-1].strftime('%d %b %y') , location


preds = pd.read_csv('https://raw.githubusercontent.com/andreapiemmi/covid-cases/main/Big_Forecasts_ESRW.csv').iloc[
    7:,:].rename(columns={'Unnamed: 0':'Date'}).set_index('Date').rename(columns={'Czechia':'Czech Republic',
                           'Moldova':'Republic of Moldova',
                           'North Macedonia':'The former Yugoslav Republic of Macedonia'})
preds.index=pd.DatetimeIndex(preds.index)

@callback(
    Output('provvi1','figure'),
    Input('Time-Slider3','value'),
    Input('EuropeMap3','clickData')
    )
def Exp_Smoo_RW(t,co_selected):
    dff=df.copy()
    dff=dff.rename(columns={'Czechia':'Czech Republic','Moldova':'Republic of Moldova',
                        'North Macedonia':'The former Yugoslav Republic of Macedonia'})
   

    if co_selected is None:
        location='Switzerland'
    else:
        location=co_selected['points'][0]['location']
    dff=preds[[*preds.columns[[location in c for c in preds.columns]]]].iloc[
        int(np.where(preds.index>dff.index[t[0]])[0][0]) : int(np.where(preds.index==dff.index[t[-1]])[0]) ,]
    fig=px.line(dff).update_layout(
        template='plotly_white',legend=dict(orientation="h",yanchor="top",y=-0.2,xanchor="left",x=0) ) 
    fig.layout.yaxis.title='Confirmed Cases'
    
    return fig


if __name__ == '__main__':
    app.run(debug=True)


