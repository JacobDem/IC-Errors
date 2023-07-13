"""
@author: Jacob Demuynck
"""
import pickle
import codecs
import base64
import io
from os.path import join
import diskcache

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, ctx, DiskcacheManager
from dash.exceptions import PreventUpdate

from IC_errors import IC_errors
import dashboard_components

DEFAULT_COLORS = px.colors.qualitative.Dark24
PANC8_DATA_FOLDER = "../data/panc8 dataset"
REJAFADA_DATA_FOLDER = "../data/rejafada dataset"

INITIAL_ICE = {}

# INITIALIZATION OF DEFAULT DATA SETS
# PANC8
indices = np.load(open(join(PANC8_DATA_FOLDER,"dataframes/panc8_indices.npy"), 'rb'), allow_pickle=True)

ice_panc8 = IC_errors(df=pd.read_pickle(join(PANC8_DATA_FOLDER,"dataframes/panc8_TSNE.pkl")), target_col='celltype', batch_col='tech')
ice_panc8.set_tsne(pd.read_pickle(join(PANC8_DATA_FOLDER,"dataframes/panc8_TSNE.pkl")))
ice_panc8.df_tsne.index = indices
ice_panc8.set_feature_bagging(np.load(open(join(PANC8_DATA_FOLDER,"results/panc8_outlier_feature_bagging"), 'rb'), allow_pickle=True))
ice_panc8.set_random_forest(np.load(open(join(PANC8_DATA_FOLDER,"results/panc8_outlier_random_forest"), 'rb'), allow_pickle=True))
ice_panc8.set_neural_training_values(np.load(open(join(PANC8_DATA_FOLDER,"results/panc8_outlier_neural_training_values"), 'rb'), allow_pickle=True))
ice_panc8.set_neural_training_switches(np.load(open(join(PANC8_DATA_FOLDER,"results/panc8_outlier_neural_training_switches"), 'rb'), allow_pickle=True))
ice_panc8.set_cleanlab(np.load(open(join(PANC8_DATA_FOLDER,"results/panc8_outlier_cleanlab"), 'rb'), allow_pickle=True))
ice_panc8.set_nearest_neighbors(np.load(open(join(PANC8_DATA_FOLDER,"results/panc8_flann_11.npy"), 'rb'), allow_pickle=True)[()])
INITIAL_ICE['panc8'] = {'ice': ice_panc8, 
                        'indices': indices,
                        'target_palette': {'ductal': 'red', 'alpha': 'orange', 'mast': 'gold', 
                                           'beta': 'olivedrab', 'macrophage': 'chartreuse', 
                                           'activated_stellate': 'blue', 'endothelial': 'darkorchid',
                                           'epsilon': 'palevioletred', 'gamma': 'brown', 
                                           'delta': 'darkgoldenrod', 'schwann': 'mediumaquamarine', 
                                           'acinar': 'fuchsia', 'quiescent_stellate': 'teal'},
                        'batch_palette': {'smartseq2': 'circle', 'indrop': 'x', 'fluidigmc1': 'cross', 
                                          'celseq': 'diamond', 'celseq2': 'square'},
}

# REJAFADA
indices = np.load(open(join(REJAFADA_DATA_FOLDER,"dataframes/rejafada_indices.npy"), 'rb'), allow_pickle=True)

ice_rejafada = IC_errors(df=pd.read_pickle(join(REJAFADA_DATA_FOLDER,"dataframes/rejafada_TSNE.pkl")), target_col='classification')
ice_rejafada.set_tsne(pd.read_pickle(join(REJAFADA_DATA_FOLDER,"dataframes/rejafada_TSNE.pkl")))
ice_rejafada.df_tsne.index = indices
ice_rejafada.set_feature_bagging(np.load(open(join(REJAFADA_DATA_FOLDER,"results/rejafada_outlier_feature_bagging"), 'rb'), allow_pickle=True))
ice_rejafada.set_random_forest(np.load(open(join(REJAFADA_DATA_FOLDER,"results/rejafada_outlier_random_forest"), 'rb'), allow_pickle=True))
ice_rejafada.set_neural_training_values(np.load(open(join(REJAFADA_DATA_FOLDER,"results/rejafada_outlier_neural_training_values"), 'rb'), allow_pickle=True))
ice_rejafada.set_neural_training_switches(np.load(open(join(REJAFADA_DATA_FOLDER,"results/rejafada_outlier_neural_training_switches"), 'rb'), allow_pickle=True))
ice_rejafada.set_cleanlab(np.load(open(join(REJAFADA_DATA_FOLDER,"results/rejafada_outlier_cleanlab"), 'rb'), allow_pickle=True))
ice_rejafada.set_nearest_neighbors(np.load(open(join(REJAFADA_DATA_FOLDER,"results/rejafada_flann_11.npy"), 'rb'), allow_pickle=True)[()])
INITIAL_ICE['rejafada'] = {'ice': ice_rejafada,
                            'indices': indices,
                            'target_palette': {'M':'red', 'B':'green'},
                            'batch_palette': None,
}

# Used for reading in CSV files
def parse_contents(contents):
    _, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')), sep=',', index_col=0)
    except Exception as exc:
        raise FileNotFoundError("Error occured while trying to parse file") from exc

    return df

# START DASH APP
cache = diskcache.Cache("./cache")
background_callback_manager   = DiskcacheManager(cache)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP])
load_figure_template('LUX')

# Default plot stylings
plot_styling = {
    'width': '140vh',
    'height': '130vh',
}

MARGIN_TOP = '0px'
MARGIN_BOTTOM = '20px'
MENU_WIDTH = 3

app.layout = html.Div(children=[
    dcc.Store('stored-ic_errors', storage_type='local', data=codecs.encode(pickle.dumps(INITIAL_ICE), "base64").decode()),
    dcc.Store('current-dataset', storage_type='local', data='panc8'),
    dcc.Store('outlier-details-visible'),
    dcc.Store('processed-dataframe'),

    dbc.Row([
        # TSNE GRAPH
        dbc.Col([
            html.H1('IC-Errors dashboard', style={'textAlign': 'center', 'padding-top':'20px', 'font-size':'50px'}),
            dcc.Graph(id='graph-TSNE',style={'height': '130vh'}),
            ],
            width=9,
        ),

        # MENU
        dbc.Col([
            dbc.Progress(value=0, striped=True, animated=True, id='progress_bar', color='lightblue'),
            html.Div([
                html.H3('Data'),
                dbc.Button('Download processed dataset', id='download-button', size='sm', style={'margin-left': 'auto'}, disabled=True),
                dcc.Download(id="download-dataframe-csv"),
            ],
            style={
                'display':'flex',
                'align-items':'center',
                'margin-top':'75px',
            }),
            html.Hr(),
            # Dataset selection
            html.Div([
                dashboard_components.preloaded_dataset_dropdown,
                dashboard_components.dataset_uploader,
            ],
            style={
                'margin-bottom': MARGIN_BOTTOM,
            }),
            
            html.Div([
                html.H3('Filtering'),
                dbc.Button('expand', id='show-outlier-details', size='sm', style={'margin-left': 'auto'}),
            ],
            style={
                'display':'flex',
                'align-items':'center',
                'margin-top':'75px',
            }),
            html.Hr(),
            # Graph marker size slider
            html.Div([
                dashboard_components.marker_size_slider,
            ],
            style={
                'margin-top': MARGIN_TOP,
                'margin-bottom': MARGIN_BOTTOM,
            }),

            # OUTLIER DETECTION
            html.Div([
                dashboard_components.general_outlier_detection,
                dashboard_components.feature_bagging,
                dashboard_components.random_forest,
                dashboard_components.neural_values,
                dashboard_components.neural_switches,
                dashboard_components.cleanlab,
                dashboard_components.intersection_toggle,
            ],
            style={
                'margin-top': MARGIN_TOP,
                'margin-bottom':MARGIN_BOTTOM,
            }),

            # LABEL ERRORS
            html.Div([
                dashboard_components.label_error,
            ],
            style={
                'margin-bottom': MARGIN_BOTTOM,
                'margin-top': MARGIN_TOP,
            }),

            # Outlier details button

        ],
        style={
            'padding-right': '25px',
            'padding-left': '25px',
            'padding-top': '50px',
            'background-color': '#F8FAFB',
        })
    ])
])

@app.callback(
    Output(component_id='current-dataset', component_property='data'),
    Output(component_id='stored-ic_errors', component_property='data'),
    Output(component_id='dataset-dropdown', component_property='options'),
    Output(component_id='dataset-dropdown', component_property='value'),
    Input(component_id='dataset-dropdown', component_property='value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State(component_id='stored-ic_errors', component_property='data'),
    background=True,
    running=[
    (
        Output("progress_bar", "style"),
        {"visibility": "visible"},
        {"visibility": "hidden"},
    ),
    ],
    manager=background_callback_manager,
    progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
)
def store_ic_errors(set_progress, chosen_dataset, list_of_contents, list_of_names, stored_ic_errors):
    stored_ic_errors = pickle.loads(codecs.decode(stored_ic_errors.encode(), "base64"))
    if chosen_dataset is None:
        chosen_dataset = 'panc8'

    if ctx.triggered_id == 'upload-data':
        if list_of_contents is not None:
            df = parse_contents(list_of_contents)
            filename = list_of_names.split('.')[0]

            target_col = df.columns[-1]
            try:
                ice = IC_errors(df,target_col)
            except Exception as exc:
                raise RuntimeError("Error occured while trying to create \
                                    an IC_errors object from the specified file") from exc

            set_progress((1, 8))

            target_palette = {}
            for i,cl in enumerate(ice.classes):
                target_palette[cl] = DEFAULT_COLORS[i]

            ice.construct_tsne()
            set_progress((2, 8))

            ice.outlier_feature_bagging(apply_threshold=False)
            set_progress((3, 8))

            ice.outlier_random_forest(apply_threshold=False)
            set_progress((4, 8))

            ice.outlier_neural_training(apply_threshold=False)
            set_progress((5, 8))

            ice.outlier_cleanlab()
            set_progress((6, 8))

            ice.nearest_neighbors()
            set_progress((7, 8))

            ice.df = ice.df_tsne
            stored_ic_errors[filename] = {'ice': ice,
                                        'indices': np.array(df.index),
                                        'target_palette': target_palette,
                                        'batch_palette': None}

            del df
            chosen_dataset = filename

            set_progress((8, 8))

    return [chosen_dataset, codecs.encode(pickle.dumps(stored_ic_errors), "base64").decode(), 
            list(stored_ic_errors.keys()), chosen_dataset]

@app.callback(
        [Output(component_id='toggle-general-outlier', component_property='on'),
         Output(component_id='toggle-feature-bagging', component_property='on'),
         Output(component_id='toggle-random-forest', component_property='on'),
         Output(component_id='toggle-neural-training-values', component_property='on'),
         Output(component_id='toggle-neural-training-switches', component_property='on'),
         Output(component_id='toggle-cleanlab', component_property='on'),
         Output(component_id='toggle-label-errors', component_property='on'),
         Output(component_id='toggle-union', component_property='value')],
         Input(component_id='dataset-dropdown', component_property='value'),
         Input('upload-data', 'contents'),
         Input(component_id='outlier-details-visible', component_property='data')
)
def reset_toggles(dataset, file, details_visible):
    return [False]*8

# Reset the click data when updating graph
@app.callback(
    Output(component_id='graph-TSNE', component_property='clickData'),
    [Input(component_id='toggle-feature-bagging', component_property='on'),
     Input(component_id='toggle-random-forest', component_property='on'),
     Input(component_id='toggle-neural-training-values', component_property='on'),
     Input(component_id='toggle-neural-training-switches', component_property='on'),
     Input(component_id='toggle-cleanlab', component_property='on'),
     Input(component_id='feature-bagging-threshold', component_property='value'),
     Input(component_id='random-forest-threshold', component_property='value'),
     Input(component_id='neural-training-values-threshold', component_property='value'),
     Input(component_id='neural-training-switches-threshold', component_property='value'),
     Input(component_id='toggle-union', component_property='value'),
     Input(component_id='outlier-details-visible', component_property='data'),
     Input(component_id='toggle-general-outlier', component_property='on'),
     Input(component_id='general-outlier-threshold', component_property='value')]
)
def reset_click_data(feature_bagging_active, random_forest_active, 
                    neural_values_active, neural_switches_active, cleanlab_active,
                    feature_bagging_threshold, random_forest_threshold,
                    neural_training_values_threshold, neural_training_switches_threshold, 
                    calculate_union, outlier_details_visible, general_outlier_active, general_outlier_threshold):
    return None


# SCATTER PLOT CALLBACK
@app.callback(
    Output(component_id='graph-TSNE', component_property='figure'),
    Output(component_id='processed-dataframe', component_property='data'),
    Output(component_id='download-button', component_property='disabled'),
    Input(component_id='current-dataset', component_property='data'),
    Input(component_id='markersize-slider', component_property='value'),
    Input(component_id='toggle-feature-bagging', component_property='on'),
    Input(component_id='toggle-random-forest', component_property='on'),
    Input(component_id='toggle-neural-training-values', component_property='on'),
    Input(component_id='toggle-neural-training-switches', component_property='on'),
    Input(component_id='toggle-cleanlab', component_property='on'),
    Input(component_id='feature-bagging-threshold', component_property='value'),
    Input(component_id='random-forest-threshold', component_property='value'),
    Input(component_id='neural-training-values-threshold', component_property='value'),
    Input(component_id='neural-training-switches-threshold', component_property='value'),
    Input(component_id='toggle-union', component_property='value'),
    Input(component_id='outlier-details-visible', component_property='data'),
    Input(component_id='toggle-general-outlier', component_property='on'),
    Input(component_id='general-outlier-threshold', component_property='value'),
    Input(component_id='graph-TSNE', component_property='clickData'),
    Input(component_id='toggle-label-errors', component_property='on'),
    Input(component_id='label-error-threshold', component_property='value'),
    State(component_id='stored-ic_errors', component_property='data')
)
def load_graph_tsne(current_dataset, marker_size, feature_bagging_active, random_forest_active, 
                    neural_values_active, neural_switches_active, cleanlab_active,
                    feature_bagging_threshold, random_forest_threshold,
                    neural_training_values_threshold, neural_training_switches_threshold, 
                    calculate_union, outlier_details_visible, general_outlier_active, general_outlier_threshold,
                    click_data, show_label_errors, label_error_threshold, stored_ic_errors):
    """ Callback function to load the t-SNE graph

    Args:
        marker_size : int
            The marker size used in the t-SNE graph
        plot_type : str
            The plot type type to show
    """
    current_dataset_decoded = pickle.loads(codecs.decode(stored_ic_errors.encode(), "base64"))[current_dataset]
    dc = current_dataset_decoded['ice']
    target_palette = current_dataset_decoded['target_palette']
    batch_palette = current_dataset_decoded['batch_palette']
    datapoint_colors = np.array([target_palette[i] for i in dc.df[dc.target_col]])
    datapoint_symbols = None if batch_palette is None else np.array([batch_palette[i] for i in dc.df[dc.batch_col]])

    # OUTLIER CALCULATIONS
    if not outlier_details_visible:
        if not general_outlier_active:
            outlier_detection_active = False
            highlight_data = np.array([True] * dc.n_samples)
        else:
            outlier_detection_active = True
            highlight_data = dc.outlier_ensemble(general_outlier_threshold)
    
    else:
        if not (feature_bagging_active or random_forest_active or 
                                        neural_switches_active or neural_values_active):
            outlier_detection_active = False
        else:
            outlier_detection_active = True

        active_outlier_detection = [not outlier_detection_active, 
                                    feature_bagging_active, random_forest_active, neural_values_active, 
                                    neural_switches_active, cleanlab_active]

        outlier_detection_highlight = np.array([np.array([True]*dc.n_samples),
                                            dc.feature_bagging_result >= feature_bagging_threshold,
                                            dc.random_forest_result <= 1 - random_forest_threshold,
                                            dc.neural_training_values_result >= neural_training_values_threshold,
                                            dc.neural_training_switches_result >= neural_training_switches_threshold,
                                            dc.cleanlab_result])
        
        if not calculate_union:
            highlight_data = np.logical_and.reduce(
                outlier_detection_highlight[active_outlier_detection])
        else:
            highlight_data = np.logical_or.reduce(
                outlier_detection_highlight[active_outlier_detection])

    fig_tsne = go.Figure()

    extra_hover_info = [""]*dc.n_samples

    # CLICKED DATA CALCULATIONS
    click_data_hover = ""
    highlight_data_index = np.where(highlight_data)[0]

    if click_data is not None and 'points' in click_data.keys():
        index_num = highlight_data_index[click_data['points'][0]['pointIndex']]
        connections = dc.nearest_neighbors_result[index_num]

        for p in connections:
            # Add nearest neighbors connections to plot
            fig_tsne.add_trace(go.Scatter(
                mode='lines',
                x=dc.df_tsne.iloc[[index_num, p]]['x'],
                y=dc.df_tsne.iloc[[index_num, p]]['y'],
                marker_color=target_palette[dc.df_tsne.iloc[p][dc.target_col]],
                hovertemplate=
                    '<b>' + dc.target_col + ': </b>' + dc.df_tsne.iloc[p][dc.target_col] + '<br>' +
                    (('<b>' + dc.batch_col + ': </b>'+ dc.df_tsne.iloc[p][dc.batch_col] + '<br>') if (dc.batch_col is not None) else ''),
                opacity=0.5,
                showlegend=False
            ))

        click_data_hover += "<br><b>Nearest neighbors:</b><br>"

        (_, neighbor_labels) = dc.get_neighbor_labels_dict(index_num)

        while len(neighbor_labels):
            max_label = max(neighbor_labels, key=neighbor_labels.get)
            click_data_hover += str(max_label) + ": " + str(neighbor_labels.pop(max_label)) + "<br>"

        extra_hover_info[index_num] = click_data_hover

    # LABEL ERRORS CALCULATIONS
    if outlier_detection_active and show_label_errors:
        (corrected_labels, corrected_labels_positions) = dc.outlier_label_correction(highlight_data, threshold=label_error_threshold)

        if len(corrected_labels_positions) != 0:
            for i in range(dc.n_samples):
                if i in corrected_labels_positions:
                    extra_hover_info[i] += '<br><b>Label error detected</b><br>Suggested label correction: ' + str(corrected_labels[i])
                else:
                    extra_hover_info[i] += '<br><b>No label error</b>'

            # Add label error highlights to plot
            fig_tsne.add_trace(go.Scatter(
                mode='markers',
                x=dc.df_tsne.iloc[corrected_labels_positions].x,
                y=dc.df_tsne.iloc[corrected_labels_positions].y,
                marker_color=[target_palette[x] for x in corrected_labels[corrected_labels_positions]],
                marker_symbol='circle',
                marker_size=marker_size*2,
                opacity=1,
                hoverinfo='skip',
                showlegend=False,
            ))
    
    extra_hover_info = np.array(extra_hover_info)

    # Add different traces to plot

    # Custom legend
    fig_tsne.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        name=dc.target_col.upper(),
        marker=dict(size=1, color='white', symbol='circle'),
    ))
    for target,color in target_palette.items():
        fig_tsne.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=target,
            marker=dict(size=7, color=color, symbol='circle'),
        ))
    if dc.batch_col is not None:
        fig_tsne.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name='',
            marker=dict(size=1, color='white', symbol='circle'),
        ))
        fig_tsne.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=dc.batch_col.upper(),
            marker=dict(size=1, color='white', symbol='circle'),
        ))
        for batch,symbol in batch_palette.items():
            fig_tsne.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=batch,
                marker=dict(size=7, color="black", symbol=symbol),
            ))
    
    # Statistics
    if outlier_detection_active:
        fig_tsne.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name='',
            marker=dict(size=1, color='white', symbol='circle'),
        ))
        fig_tsne.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name='',
            marker=dict(size=1, color='white', symbol='circle'),
        ))
        fig_tsne.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name='OUTLIERS: ' + str(round(len(np.where(highlight_data)[0])/dc.n_samples * 100, 2)) + '%',
            marker=dict(size=1, color='white', symbol='circle'),
        ))
        if show_label_errors:
            fig_tsne.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name='LABEL ERRORS: ' + str(round(len(corrected_labels_positions)/dc.n_samples * 100, 2)) + '%',
            marker=dict(size=1, color='white', symbol='circle'),
        ))


    # Non-outlier points
    fig_tsne.add_trace(go.Scatter(
        name='Non-outlier' if outlier_detection_active else '',
        mode='markers',
        x=dc.df_tsne[~highlight_data].x,
        y=dc.df_tsne[~highlight_data].y,
        marker_color=datapoint_colors[~highlight_data],
        marker_symbol='circle' if (datapoint_symbols is None) else datapoint_symbols[~highlight_data],
        marker_size=marker_size,
        opacity=0.2,
        hoverinfo='skip',
        showlegend=False
    ))

    # Outlier points
    fig_tsne.add_trace(go.Scatter(
        name='Outlier' if outlier_detection_active else '',
        mode='markers',
        x=dc.df_tsne[highlight_data].x,
        y=dc.df_tsne[highlight_data].y,
        customdata=np.stack((np.array(dc.df_tsne.index)[highlight_data], dc.df_tsne[dc.target_col][highlight_data], extra_hover_info[highlight_data]), 
                             axis=-1) if (dc.batch_col is None) else np.stack((np.array(dc.df_tsne.index)[highlight_data], 
                                                                               dc.df_tsne[dc.target_col][highlight_data], 
                                                                               dc.df_tsne[dc.batch_col][highlight_data], 
                                                                                extra_hover_info[highlight_data]), 
                                                                                axis=-1), 
        hovertemplate=
            '<b>' + '%{customdata[0]} </b> <br>' +
            '<b>' + dc.target_col + ': </b> %{customdata[1]} <br>' +
            (('<b>' + dc.batch_col + ': </b> %{customdata[2]} <br>') if (dc.batch_col is not None) else '') +
            ('%{customdata[3]}' if (dc.batch_col is not None) else '%{customdata[2]}'),
        marker_color=datapoint_colors[highlight_data],
        marker_symbol='circle' if (datapoint_symbols is None) else datapoint_symbols[highlight_data],
        marker_size=marker_size,
        opacity=1,
        showlegend=False
    ))

    fig_tsne.update_layout(legend={'itemsizing': 'constant'}, uirevision='constant')

    df_processed = pd.DataFrame()
    df_processed.index = dc.df_tsne.index
    if outlier_detection_active:
        df_processed['outlier'] = highlight_data
        if show_label_errors:
            corrected_labels_boolean = [True if x in corrected_labels_positions else False for x in range(dc.n_samples)]
            df_processed['label error'] = corrected_labels_boolean
            df_processed['correct labels'] = corrected_labels

    return [fig_tsne, codecs.encode(pickle.dumps(df_processed), "base64").decode(), not outlier_detection_active]

@app.callback(
    [Output(component_id='outlier-details-visible', component_property='data'),
    Output(component_id='show-outlier-details', component_property='children')],
    Input(component_id='show-outlier-details', component_property='n_clicks')
)
def toggle_outlier_details(n_clicks):
    if not n_clicks or n_clicks % 2 == 0:
        return [False, 'expand']
    return [True, 'hide details']

@app.callback(
    [Output(component_id='feature-bagging-options-div', component_property='hidden'),
     Output(component_id='random-forest-options-div', component_property='hidden'),
     Output(component_id='neural-training-values-options-div', component_property='hidden'),
     Output(component_id='neural-training-switches-options-div', component_property='hidden'),
     Output(component_id='cleanlab-options-div', component_property='hidden'),
     Output(component_id='union-div', component_property='hidden'),
     Output(component_id='general-outlier-options-div', component_property='hidden')],
    Input(component_id='outlier-details-visible', component_property='data')
)
def toggle_options_divs(outlier_details_visible):
    return [not outlier_details_visible] * 6 + [outlier_details_visible]

@app.callback(
    Output(component_id='toggle-label-errors-div', component_property='hidden'),
    [Input(component_id='toggle-feature-bagging', component_property='on'),
    Input(component_id='toggle-random-forest', component_property='on'),
    Input(component_id='toggle-neural-training-values', component_property='on'),
    Input(component_id='toggle-neural-training-switches', component_property='on'),
    Input(component_id='toggle-cleanlab', component_property='on'),
    Input(component_id='toggle-general-outlier', component_property='on')]
)
def toggle_label_errors(feature_bagging_active, random_forest_active, 
                    neural_values_active, neural_switches_active, cleanlab_active, general_outlier_active):
    return not (feature_bagging_active or random_forest_active or neural_values_active or 
                neural_switches_active or cleanlab_active or general_outlier_active)

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    State(component_id='processed-dataframe', component_property='data'),
    prevent_initial_call=True,
)
def func(n_clicks, df_pickled):
    df_processed = pickle.loads(codecs.decode(df_pickled.encode(), "base64"))
    return dcc.send_data_frame(df_processed.to_csv, "data_processed.csv")

# if __name__ == "__main__":
#     app.run_server(debug=True)
