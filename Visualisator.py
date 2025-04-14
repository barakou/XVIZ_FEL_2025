import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns

# Načtení dat
df = pd.read_csv('patient_data.csv')

# Přidání textových popisků sportovní kategorie a četnosti sportování
sport_category_map = {0: "Rychlostně-silové sporty", 1: "Vytrvalostní sporty", 2: "Sportovní hry"}
df['Sport_Category_Name'] = df['Sport_Category'].map(sport_category_map)

physical_load_map = {0: "Rekreační", 1: "Závodní"}
activity_frequency_map = {0: "Nepravidelné", 1: "Pravidelné"}

df['Physical_Load_Label'] = df['Physical_Load'].map(physical_load_map)
df['Activity_Frequency_Label'] = df['Activity_Frequency'].map(activity_frequency_map)

# Výpočet přesnosti pro 5A (Euklidovská vzdálenost)
for attempt in range(1, 4):
    df[f'5A_Accuracy_{attempt}'] = np.sqrt(
        (df[f'5A_End_X_{attempt}'] - df[f'5A_Target_X_{attempt}']) ** 2 +
        (df[f'5A_End_Y_{attempt}'] - df[f'5A_Target_Y_{attempt}']) ** 2
    )

# Inicializace aplikace Dash
app = dash.Dash(__name__)

def create_filters(station_id):
    return  html.Div([
                html.Label("Vyber pokus:"),
                dcc.Dropdown(
                    id=f'attempt-selector-{station_id}',
                    options=[{'label': f'Pokus {i}', 'value': i} for i in range(1, 4)],
                    value=1
                ),

                html.Label("Filtruj podle věku:"),
                dcc.RangeSlider(
                    id=f'age-slider-{station_id}',
                    min=df['Age'].min(),
                    max=df['Age'].max(),
                    value=[df['Age'].min(), df['Age'].max()],
                    marks={i: str(i) for i in range(df['Age'].min(), df['Age'].max() + 1, 2)},
                    step=1
                ),

                html.Label("Filtruj podle pohlaví:"),
                dcc.Dropdown(
                    id=f'gender-selector-{station_id}',
                    options=[
                        {'label': 'Muž', 'value': 'Male'},
                        {'label': 'Žena', 'value': 'Female'}
                    ],
                    value=None,
                    multi=True
                ),

                html.Label("Sportovní kategorie:"),
                dcc.Dropdown(
                    id=f'sport-category-selector-{station_id}',
                    options=[
                        {'label': name, 'value': key} for key, name in sport_category_map.items()
                    ],
                    value=None,
                    multi=True
                ),

                html.Label("Fyzické zatížení:"),
                dcc.Dropdown(
                    id=f'physical-load-selector-{station_id}',
                    options=[
                        {'label': name, 'value': key} for key, name in physical_load_map.items()
                    ],
                    value=None,
                    multi=True
                ),

                html.Label("Pravidelnost sportování:"),
                dcc.Dropdown(
                    id=f'activity-frequency-selector-{station_id}',
                    options=[
                        {'label': name, 'value': key} for key, name in activity_frequency_map.items()
                    ],
                    value=None,
                    multi=True
                ),

                html.Label("Vyber konkrétní sport:"),
                dcc.Dropdown(
                    id=f'sport-selector-{station_id}',
                    options=[{'label': sport, 'value': sport} for sport in df['Sport'].unique()],
                    value=None,
                    multi=True
                ),
            ], style={'width': '50%', 'display': 'inline-block'})


# Rozložení aplikace
app.layout = html.Div([
    dcc.Tabs([
        # Stanoviště 1A - Výška, velikost chodidla, délka boků
        dcc.Tab(label='Stanoviště 1A', children=[
            create_filters("1A"),
            dcc.Graph(id='height-distribution-1A'),
            dcc.Graph(id='foot-size-distribution-1A'),
            dcc.Graph(id='hips-length-distribution-1A')
        ]),

        # Stanoviště 2B - Odchylky vzdáleností
        dcc.Tab(label='Stanoviště 2B', children=[
            create_filters("2B"),
            dcc.Graph(id='2B-histogram-distance1'),
            dcc.Graph(id='2B-histogram-distance2'),
            #dcc.Graph(id='2B-overlay-histogram'),
            dcc.Graph(id='2B-scatter-distance1-distance2'),
            #dcc.Graph(id='2B-arrow-plot')
        ]),

        # Stanoviště 2C - Rozpoznání směru pohybu
        dcc.Tab(label='Stanoviště 2C', children=[
            create_filters("2C"),
            dcc.Graph(id='2C-response-time'),
            dcc.Graph(id='2C-correctness')
        ]),

        # Stanoviště 3A - Stereognozie šíře
        dcc.Tab(label='Stanoviště 3A', children=[
            create_filters("3A"),
            dcc.Graph(id = '3A-estimated-width'),
            dcc.Graph(id = '3A-deviation')
        ]),

        # Stanoviště 3B - Test váhy
        dcc.Tab(label='Stanoviště 3B', children=[
            create_filters("3B"),
            dcc.Graph(id='3B-weight-choices'),
            dcc.Graph(id='3B-weight-chois')
        ]),

        # Stanoviště 3C - Stereognozie tvar
        dcc.Tab(label='Stanoviště 3C', children=[
            create_filters("3C"),
            dcc.Graph(id='3C-barplot-correctness'),
            dcc.Graph(id='3C-heatmap-distribution'),
            dcc.Graph(id='3C-confusion-matrix')
        ]),

        # Stanoviště 4A - Propriocepce
        dcc.Tab(label='Stanoviště 4A', children=[
            create_filters("4A"),
            dcc.Graph(id='4A-correctness'),
            dcc.Graph(id='4A-correctnss')
        ]),

        # Stanoviště 5A - Orientace
        dcc.Tab(label='Stanoviště 5A', children=[
            create_filters("5A"),
            dcc.Graph(id='scatter-accuracy-age-5A'),
            dcc.Graph(id='scatter-matrix-5A'),
            dcc.Graph(id='box-accuracy-sport-5A'),
            dcc.Graph(id='scatter-endpoints-5A')
        ]),

        # Stanoviště 7A - Reakce na sílu
        dcc.Tab(label='Stanoviště 7A', children=[
            create_filters("7A"),
            dcc.Graph(id='7A-trend-display'),
            dcc.Graph(id='7A-horizon-chart'),
            dcc.Graph(id='7A-scatter-smoothness'),
            dcc.Graph(id='7A-force-vs-position'),
        ])
    ])
])
# Callback pro dynamické filtrování dat ve stanovišti 1A
@app.callback(
    [Output('height-distribution-1A', 'figure'),
     Output('foot-size-distribution-1A', 'figure'),
     Output('hips-length-distribution-1A', 'figure')],
    [Input('attempt-selector-1A', 'value'),
     Input('age-slider-1A', 'value'),
     Input('gender-selector-1A', 'value'),
     Input('sport-category-selector-1A', 'value'),
     Input('physical-load-selector-1A', 'value'),
     Input('activity-frequency-selector-1A', 'value'),
     Input('sport-selector-1A', 'value')]
)
def update_1A_graphs(selected_attempt, age_range, gender, sport_category, physical_load, activity_frequency, sport):
    # Filtrace dat podle vybraných kritérií
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])
    ]
    if gender:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]
    if sport_category:
        filtered_df = filtered_df[filtered_df['Sport_Category'].isin(sport_category)]
    if physical_load:
        filtered_df = filtered_df[filtered_df['Physical_Load'].isin(physical_load)]
    if activity_frequency:
        filtered_df = filtered_df[filtered_df['Activity_Frequency'].isin(activity_frequency)]
    if sport:
        filtered_df = filtered_df[filtered_df['Sport'].isin(sport)]

    height_histogram = px.histogram(filtered_df, x='1A_Height', title='Rozložení výšky')
    foot_histogram = px.histogram(filtered_df, x='1A_Foot_Size', title='Rozložení velikosti chodidla')
    hips_histogram = px.histogram(filtered_df, x='1A_Hips_Length', title='Rozložení délky boků')

    return height_histogram, foot_histogram, hips_histogram

# Callback pro dynamické filtrování dat ve stanovišti 2B
@app.callback(
    [Output('2B-histogram-distance1', 'figure'),
     Output('2B-histogram-distance2', 'figure'),
     #Output('2B-overlay-histogram', 'figure'),
     Output('2B-scatter-distance1-distance2', 'figure')],
     #Output('2B-arrow-plot', 'figure')],
    [Input('attempt-selector-2B', 'value'),
     Input(f'age-slider-2B', 'value'),
     Input(f'gender-selector-2B', 'value'),
     Input(f'sport-category-selector-2B', 'value'),
     Input('physical-load-selector-2B', 'value'),
     Input('activity-frequency-selector-2B', 'value'),
     Input('sport-selector-2B', 'value')]
)
def update_2B_graphs(selected_attempt, age_range, gender, sport_category, physical_load, activity_frequency, sport):
    # Filtrace dat podle vybraných kritérií
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])
    ]
    if gender:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]
    if sport_category:
        filtered_df = filtered_df[filtered_df['Sport_Category'].isin(sport_category)]
    if physical_load:
        filtered_df = filtered_df[filtered_df['Physical_Load'].isin(physical_load)]
    if activity_frequency:
        filtered_df = filtered_df[filtered_df['Activity_Frequency'].isin(activity_frequency)]
    if sport:
        filtered_df = filtered_df[filtered_df['Sport'].isin(sport)]

    fig_distance1 = px.histogram(filtered_df, x='2B_Distance_Deviation1_1', title='Histogram prvního dotyku')
    fig_distance2 = px.histogram(filtered_df, x='2B_Distance_Deviation2_1', title='Histogram druhého dotyku')

    # overlay_fig = go.Figure()
    # overlay_fig.add_trace(go.Histogram(x=filtered_df['2B_Distance_Deviation1_1'], name='První dotek', opacity=0.6))
    # overlay_fig.add_trace(go.Histogram(x=filtered_df['2B_Distance_Deviation2_1'], name='Druhý dotek', opacity=0.6))
    # overlay_fig.update_layout(title='Overlay histogram prvního a druhého doteku')

    scatter_fig = px.scatter(
        filtered_df, x='2B_Distance_Deviation1_1', y='2B_Distance_Deviation2_1', color='Age_Group',
        title='Scatterplot Distance1 vs. Distance2'
    )

    # arrow_fig = go.Figure()
    # for _, row in filtered_df.iterrows():
    #     arrow_fig.add_trace(go.Scatter(
    #         x=[row['2B_Distance_Deviation1_1'], row['2B_Distance_Deviation1_3']],
    #         y=[row['2B_Distance_Deviation2_1'], row['2B_Distance_Deviation2_3']],
    #         mode='lines+markers',
    #         marker=dict(size=[8, 12], symbol=['circle', 'triangle-up']),  # Kruh a šipka
    #         line=dict(width=2),
    #         name='Pohyb ruky'
    #     ))
    #
    # arrow_fig.update_layout(title='Arrow plot - směr pohybu ruky')
    #
    # return fig_distance1, fig_distance2, overlay_fig, scatter_fig, arrow_fig
    return fig_distance1, fig_distance2, scatter_fig

# Callback pro dynamické filtrování dat ve stanovišti 2C
@app.callback(
    [Output('2C-response-time', 'figure'),
     Output('2C-correctness', 'figure')],
    [Input('attempt-selector-2C', 'value'),
     Input('age-slider-2C', 'value'),
     Input('gender-selector-2C', 'value'),
     Input('sport-category-selector-2C', 'value'),
     Input('physical-load-selector-2C', 'value'),
     Input('activity-frequency-selector-2C', 'value'),
     Input('sport-selector-2C', 'value')]
)
def update_2C_graphs(selected_attempt, age_range, gender, sport_category, physical_load, activity_frequency, sport):
    # Filtrace dat podle vybraných kritérií
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])
    ]
    if gender:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]
    if sport_category:
        filtered_df = filtered_df[filtered_df['Sport_Category'].isin(sport_category)]
    if physical_load:
        filtered_df = filtered_df[filtered_df['Physical_Load'].isin(physical_load)]
    if activity_frequency:
        filtered_df = filtered_df[filtered_df['Activity_Frequency'].isin(activity_frequency)]
    if sport:
        filtered_df = filtered_df[filtered_df['Sport'].isin(sport)]

    ans1_boxplot = px.box(filtered_df, x='2C_Answer_Time_1', title='Čas odpovědi v 2C')
    ans2_boxplot = px.histogram(filtered_df, x='2C_Answer_1', title='Úspěšnost odpovědí v 2C')

    return ans1_boxplot, ans2_boxplot

# Callback pro dynamické filtrování dat ve stanovišti 3A
@app.callback(
    [Output('3A-estimated-width', 'figure'),
     Output('3A-deviation', 'figure')],
    [Input('attempt-selector-3A', 'value'),
     Input('age-slider-3A', 'value'),
     Input('gender-selector-3A', 'value'),
     Input('sport-category-selector-3A', 'value'),
     Input('physical-load-selector-3A', 'value'),
     Input('activity-frequency-selector-3A', 'value'),
     Input('sport-selector-3A', 'value')]
)
def update_3A_graphs(selected_attempt, age_range, gender, sport_category, physical_load, activity_frequency, sport):
    # Filtrace dat podle vybraných kritérií
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])
    ]
    if gender:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]
    if sport_category:
        filtered_df = filtered_df[filtered_df['Sport_Category'].isin(sport_category)]
    if physical_load:
        filtered_df = filtered_df[filtered_df['Physical_Load'].isin(physical_load)]
    if activity_frequency:
        filtered_df = filtered_df[filtered_df['Activity_Frequency'].isin(activity_frequency)]
    if sport:
        filtered_df = filtered_df[filtered_df['Sport'].isin(sport)]

    width = px.box(filtered_df, x='3A_Estimated_Width_1', title='Odhadovaná šířka objektu')
    deviation = px.scatter(filtered_df, x='3A_Lower_X_Deviation_1', y='3A_Upper_X_Deviation_1', title='Odchylky při měření šířky')

    return width, deviation

# Callback pro dynamické filtrování dat ve stanovišti 3B
@app.callback(
    [Output('3B-weight-choices', 'figure'),
     Output('3B-weight-chois', 'figure')],
    [Input('attempt-selector-3B', 'value'),
     Input('age-slider-3B', 'value'),
     Input('gender-selector-3B', 'value'),
     Input('sport-category-selector-3B', 'value'),
     Input('physical-load-selector-3B', 'value'),
     Input('activity-frequency-selector-3B', 'value'),
     Input('sport-selector-3B', 'value')]
)
def update_3B_graphs(selected_attempt, age_range, gender, sport_category, physical_load, activity_frequency, sport):
    # Filtrace dat podle vybraných kritérií
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])
    ]
    if gender:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]
    if sport_category:
        filtered_df = filtered_df[filtered_df['Sport_Category'].isin(sport_category)]
    if physical_load:
        filtered_df = filtered_df[filtered_df['Physical_Load'].isin(physical_load)]
    if activity_frequency:
        filtered_df = filtered_df[filtered_df['Activity_Frequency'].isin(activity_frequency)]
    if sport:
        filtered_df = filtered_df[filtered_df['Sport'].isin(sport)]

    mm = px.histogram(filtered_df, x='3B_Choice_Done_1', title='Výsledky výběru těžšího objektu')
    mm2 = px.histogram(filtered_df, x='3B_Choice_Done_1', title='Výsledky výběru těžšího objektu')

    return mm,mm2

# Callback pro dynamické filtrování dat ve stanovišti 3C
@app.callback(
    [Output('3C-barplot-correctness', 'figure'),
     Output('3C-heatmap-distribution', 'figure'),
     Output('3C-confusion-matrix', 'figure')],
    [Input('attempt-selector-3C', 'value'),
     Input('age-slider-3C', 'value'),
     Input('gender-selector-3C', 'value'),
     Input('sport-category-selector-3C', 'value'),
     Input('physical-load-selector-3C', 'value'),
     Input('activity-frequency-selector-3C', 'value'),
     Input('sport-selector-3C', 'value')]
)

def update_3C_graphs(selected_attempt, age_range, gender, sport_category, physical_load, activity_frequency, sport):
    # Filtrace dat podle vybraných kritérií
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])
    ]
    if gender:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]
    if sport_category:
        filtered_df = filtered_df[filtered_df['Sport_Category'].isin(sport_category)]
    if physical_load:
        filtered_df = filtered_df[filtered_df['Physical_Load'].isin(physical_load)]
    if activity_frequency:
        filtered_df = filtered_df[filtered_df['Activity_Frequency'].isin(activity_frequency)]
    if sport:
        filtered_df = filtered_df[filtered_df['Sport'].isin(sport)]

    # Přidání sloupce pro správnost odpovědi
    filtered_df['Correct'] = (filtered_df['3C_Gesture_Displayed_1'] == filtered_df['3C_Gesture_Selected_1']).astype(int)

    #  **Barplot správných a chybných odpovědí**
    barplot_fig = px.bar(
        filtered_df.groupby('3C_Gesture_Displayed_1')['Correct'].mean().reset_index(),
        x='3C_Gesture_Displayed_1', y='Correct',
        title='Úspěšnost rozpoznání jednotlivých tvarů',
        labels={'gestureDisplayed': 'Zobrazený tvar', 'Correct': 'Podíl správných odpovědí'},
        color='Correct'
    )

    # **Heatmapa distribuce výběrů v blackboxu**
    heatmap_data = filtered_df.groupby(['3C_Gesture_Displayed_1', '3C_Selected_Position_1']).size().unstack(fill_value=0)
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis'
    ))
    heatmap_fig.update_layout(
        title='Distribuce výběrů pozic v blackboxu',
        xaxis_title='Pozice výběru',
        yaxis_title='Zobrazený tvar'
    )

    # **Matice záměn**
    confusion_matrix = pd.crosstab(filtered_df['3C_Gesture_Displayed_1'], filtered_df['3C_Gesture_Selected_1'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Matice záměn (confusion matrix)")
    ax.set_xlabel("Vybraný tvar")
    ax.set_ylabel("Zobrazený tvar")

    # Uložení obrázku a jeho načtení do grafu Plotly
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    confusion_fig = px.imshow(plt.imread("confusion_matrix.png"), title="Matice záměn")

    return barplot_fig, heatmap_fig, confusion_fig
# Callback pro dynamické filtrování dat ve stanovišti 4A
@app.callback(
    [Output('4A-correctness', 'figure'),
     Output('4A-correctnss', 'figure')],
    [Input('attempt-selector-4A', 'value'),
     Input('age-slider-4A', 'value'),
     Input('gender-selector-4A', 'value'),
     Input('sport-category-selector-4A', 'value'),
     Input('physical-load-selector-4A', 'value'),
     Input('activity-frequency-selector-4A', 'value'),
     Input('sport-selector-4A', 'value')]
)
def update_4A_graphs(selected_attempt, age_range, gender, sport_category, physical_load, activity_frequency, sport):
    # Filtrace dat podle vybraných kritérií
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])
    ]
    if gender:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]
    if sport_category:
        filtered_df = filtered_df[filtered_df['Sport_Category'].isin(sport_category)]
    if physical_load:
        filtered_df = filtered_df[filtered_df['Physical_Load'].isin(physical_load)]
    if activity_frequency:
        filtered_df = filtered_df[filtered_df['Activity_Frequency'].isin(activity_frequency)]
    if sport:
        filtered_df = filtered_df[filtered_df['Sport'].isin(sport)]

    figure = figure=px.histogram(filtered_df, x='4A_Gesture_Selected_1', title='Správnost výběru gest')
    figure2 = figure=px.histogram(filtered_df, x='4A_Gesture_Selected_1', title='Správnost výběru gest')

    return figure, figure2

# Callback pro dynamické filtrování dat ve stanovišti 5A
@app.callback(
    [Output('scatter-accuracy-age-5A', 'figure'),
     Output('scatter-matrix-5A', 'figure'),
     Output('box-accuracy-sport-5A', 'figure'),
     Output('scatter-endpoints-5A', 'figure')],
    [Input('attempt-selector-5A', 'value'),
     Input('age-slider-5A', 'value'),
     Input('gender-selector-5A', 'value'),
     Input('sport-category-selector-5A', 'value'),
     Input('physical-load-selector-5A', 'value'),
     Input('activity-frequency-selector-5A', 'value'),
     Input('sport-selector-5A', 'value')]
)
def update_5A_graphs(selected_attempt, age_range, gender, sport_category, physical_load, activity_frequency, sport):
    # Filtrace dat podle vybraných kritérií
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])
    ]
    if gender:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]
    if sport_category:
        filtered_df = filtered_df[filtered_df['Sport_Category'].isin(sport_category)]
    if physical_load:
        filtered_df = filtered_df[filtered_df['Physical_Load'].isin(physical_load)]
    if activity_frequency:
        filtered_df = filtered_df[filtered_df['Activity_Frequency'].isin(activity_frequency)]
    if sport:
        filtered_df = filtered_df[filtered_df['Sport'].isin(sport)]

    # Scatter plot přesnosti vs. věk
    scatter_fig = px.scatter(
        filtered_df, x='Age', y=f'5A_Accuracy_{selected_attempt}', color='Age_Group',
        title=f'Přesnost vs. Věk (5A - Pokus {selected_attempt})',
        labels={'Age': 'Věk', f'5A_Accuracy_{selected_attempt}': 'Přesnost'}
    )

    # Výběr relevantních sloupců pro scatterplot matici
    scatter_vars = ['Age', 'Height', 'Foot_Size', f'5A_End_X_{selected_attempt}', f'5A_End_Y_{selected_attempt}',
                    f'5A_Accuracy_{selected_attempt}']

    # Scatterplot matice
    scatter_matrix_fig = px.scatter_matrix(
        df, dimensions=scatter_vars, color='Sport_Category_Name',
        title=f'Scatterplot Matice pro Pokus {selected_attempt}',
        labels={var: var.replace("_", " ") for var in scatter_vars},
        height=800
    )

    # Box plot přesnosti podle sportovní kategorie a frekvence aktivity
    box_fig = px.box(
        filtered_df, x='Sport_Category_Name', y=f'5A_Accuracy_{selected_attempt}', color='Activity_Frequency_Label',
        title=f'Přesnost podle Sportovní Kategorie a Aktivity (5A - Pokus {selected_attempt})',
        labels={'Sport_Category_Name': 'Kategorie Sportu', f'5A_Accuracy_{selected_attempt}': 'Přesnost'}
    )

    # Scatter plot: Kam se testovaní dostali
    scatter_endpoints = px.scatter(
        filtered_df, x=f'5A_End_X_{selected_attempt}', y=f'5A_End_Y_{selected_attempt}', color='Sport_Category_Name',
        title=f'Kam se testovaní dostali (5A - Pokus {selected_attempt})',
        labels={f'5A_End_X_{selected_attempt}': 'End X', f'5A_End_Y_{selected_attempt}': 'End Y'},
        opacity=0.7
    )

    # Přidání cílového bodu (Target_X, Target_Y)
    target_x = df[f'5A_Target_X_{selected_attempt}'].iloc[0]
    target_y = df[f'5A_Target_Y_{selected_attempt}'].iloc[0]

    scatter_endpoints.add_trace(go.Scatter(
        x=[target_x], y=[target_y],
        mode='markers+text',
        marker=dict(color='red', size=15, symbol='star'),
        text=['Cílový bod'],
        textposition='top center'
    ))

    return scatter_fig, scatter_matrix_fig, box_fig, scatter_endpoints

# Callback pro dynamické filtrování dat ve stanovišti 7A
@app.callback(
    [Output('7A-trend-display', 'figure'),
     Output('7A-horizon-chart', 'figure'),
     Output('7A-scatter-smoothness', 'figure'),
     Output('7A-force-vs-position', 'figure')],
    [Input('attempt-selector-7A', 'value'),
     Input('age-slider-7A', 'value'),
     Input('gender-selector-7A', 'value'),
     Input('sport-category-selector-7A', 'value'),
     Input('physical-load-selector-7A', 'value'),
     Input('activity-frequency-selector-7A', 'value'),
     Input('sport-selector-7A', 'value')]
)

def update_7A_graphs(selected_attempt, age_range, gender, sport_category, physical_load, activity_frequency, sport):
    # Filtrace dat podle vybraných kritérií
    filtered_df = df[
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])
    ]
    if gender:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender)]
    if sport_category:
        filtered_df = filtered_df[filtered_df['Sport_Category'].isin(sport_category)]
    if physical_load:
        filtered_df = filtered_df[filtered_df['Physical_Load'].isin(physical_load)]
    if activity_frequency:
        filtered_df = filtered_df[filtered_df['Activity_Frequency'].isin(activity_frequency)]
    if sport:
        filtered_df = filtered_df[filtered_df['Sport'].isin(sport)]

        # **Trend Display: surová data**
    trend_display_fig = go.Figure()
    trend_display_fig.add_trace(go.Scatter(
        x=filtered_df['7A_Continuity_Time'],
        y=filtered_df['7A_Movement_Smoothness'],
        mode='lines',
        name='Poloha v čase'
    ))
    trend_display_fig.add_trace(go.Scatter(
        x=filtered_df['7A_Continuity_Time'],
        y=filtered_df['7A_Continuity_Mean_Torque'],
        mode='lines',
        name='Síla (7A_Continuity_Mean_Torque)'
    ))
    trend_display_fig.update_layout(title="Trend Display - Poloha a Síla v čase")

    # **Horizon Chart pro vizualizaci oscilací pohybu**
    # Fourierova transformace pro analýzu oscilací pohybu
    N = len(filtered_df)
    T = 0.01  # Interval vzorkování (10 ms)
    yf = fft(filtered_df['7A_Movement_Smoothness'].values)
    xf = fftfreq(N, T)[:N // 2]

    horizon_chart_fig = go.Figure()
    horizon_chart_fig.add_trace(go.Scatter(
        x=xf,
        y=2.0 / N * np.abs(yf[:N // 2]),
        mode='lines',
        name='Oscilace (FFT)'
    ))
    horizon_chart_fig.update_layout(title="Horizon Chart - Oscilace pohybu")

    #  **Scatter plot pro plynulost pohybu**
    scatter_smoothness_fig = px.scatter(
        filtered_df, x='7A_Continuity_Mean_Torque', y='7A_Movement_Smoothness',
        title='Scatter Plot - Plynulost pohybu vs. Síla'
    )

    # **Vizualizace reakce na sílu (7A_Continuity_Mean_Torque vs. 7A_Movement_Smoothness vs. Time)**
    force_vs_position_fig = go.Figure()
    force_vs_position_fig.add_trace(go.Scatter3d(
        x=filtered_df['7A_Continuity_Time'],
        y=filtered_df['7A_Movement_Smoothness'],
        z=filtered_df['7A_Continuity_Mean_Torque'],
        mode='lines',
        marker=dict(size=5, color=filtered_df['7A_Continuity_Mean_Torque'], colorscale='Viridis'),
        name="Síla vs. Poloha vs. Čas"
    ))
    force_vs_position_fig.update_layout(
        title="3D Vizualizace: Síla vs. Poloha vs. Čas",
        scene=dict(
            xaxis_title='Čas',
            yaxis_title='Poloha',
            zaxis_title='Síla (7A_Continuity_Mean_Torque)'
        )
    )

    return trend_display_fig, horizon_chart_fig, scatter_smoothness_fig, force_vs_position_fig

# Spuštění aplikace
if __name__ == '__main__':
    app.run_server(debug=True)