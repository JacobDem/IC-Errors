from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_daq as daq

COMPONENT_TOP_MARGIN = '30px'
COMPONENT_BOTTOM_MARGIN = '10px'

DATASET_UPLOADER_TEXT = \
    "The IC-Errors dashboard currently only accepts .csv files. \
    \n The file should meet the following requirements:\
    \n \n • The file uses a comma (,) as separator\
    \n \n • All columns are numerical, except for the target classification column\
    \n \n • The target classification column is the last column in the csv file \
    \n \n • The first row is a header row and the first column contains the row indices"

GENERAL_OUTLIER_TEXT = \
    "The outlier detection algorithm is a combination of four different outlier methods (see the expanded details for more info). \
    From these methods, the intersection of outliers is taken and highlighted in the plot. \
    \n \n The threshold slider below indicates the percentage of data points that each individual outlier method will mark as outlier, before the intersection is taken."

FEATURE_BAGGING_TEXT = \
    "The \"feature bagging\" outlier algorithm uses multiple rounds of the Local Outlier Factor algorithm, \
    with every round working on a different, random subset of attributes. \n \n \
    The threshold below indicates the minimum percentage of rounds in which a datapoint needs to be flagged as outlier, \
    to be highlighted as an actual outlier in the plot."

RANDOM_FOREST_TEXT = \
    "The \"random forest\" outlier algorithm trains a Random Forest classifier on the dataset, \
    and for every datapoint looks at the classifier's prediction probability of that point's true label. \n \n \
    If the classifier's prediction probability is lower than 1 - the threshold, that data point will be marked as outlier."

NEURAL_VALUES_TEXT = \
    "The \"neural training\" outlier algorithm trains a simple Neural network on the dataset in a set amount of epochs, \
    while examining the training behavior. This particual methods inspects the amount of different prediction values \
    a datapoint is classified as during the training. \n \n The threshold below indicates the percentage of the number of \
    unique class labels that a data point should be classified as at least once during training to be considered an outlier."

NEURAL_SWITCHES_TEXT = \
    "The \"neural training\" outlier algorithm trains a simple neural network on the dataset in a set amount of epochs, \
    while examining the training behavior. This particual methods inspects the amount of times the predicted class of \
    a datapoint switches during training. \n \n The threshold below indicates \
    the percentage of times that a data point's prediction needs to switch to be considered an outlier."

CLEANLAB_TEXT = \
    "This \"label issue\" outlier algorithm was developed by Cleanlab, and uses \"confident learning\" to \
    detect noisy data. It uses a pre-trained random forest classifier and inspects the entropy of a datapoint's \
    prediction probabilities. More information can be found on the CleanLab website."

LABEL_ERROR_TEXT = \
    "The \"label error detection\" algorithm calculates the 10 nearest neighbors in high-dimensional space of all outliers. \
    A data point is marked as having a label error, if none of its 10 neighbors has the same label, and if a certain fraction of its \
    neighbors have the same label. This fraction is indicated by the threshold below."

preloaded_dataset_dropdown = html.Div([
    html.B('Choose a pre-loaded data set:'),
    dcc.Dropdown(id='dataset-dropdown', clearable=False, searchable=False)
    ],
    style={
        'margin-top': '10px',
        'margin-bottom':COMPONENT_BOTTOM_MARGIN,
})

dataset_uploader = html.Div([
    html.B('Upload a custom data set:'),
    html.I(
        className="bi bi-info-circle-fill",
        id='data-upload-info-icon',
        style={
            'margin-left':'5px',
            'margin-right':'15px'
        }
    ),
    html.P('Before uploading, make sure your data set complies with the required data structure', style={'font-size':'small'}),
    dbc.Tooltip(dcc.Markdown(DATASET_UPLOADER_TEXT),
        target="data-upload-info-icon",
    ),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False,
    )],
    style={
        'margin-top': COMPONENT_TOP_MARGIN,
        'margin-bottom':COMPONENT_BOTTOM_MARGIN,
    }
)

marker_size_slider = html.Div([
    html.B('Marker sizes'),
    dcc.Slider(1, 10, 1,
                value=4,
                id='markersize-slider',
                ),
    ],
    style={
        'margin-top': '20px',
        'margin-bottom': COMPONENT_BOTTOM_MARGIN,
})

general_outlier_detection = html.Div([
    html.B('Outlier detection'),
    html.I(
        className="bi bi-info-circle-fill",
        id='general-outlier-info-icon',
        style={
            'margin-left':'5px',
            'margin-right':'15px'
        }
    ),
    dbc.Tooltip(dcc.Markdown(GENERAL_OUTLIER_TEXT),
        target="general-outlier-info-icon",
    ),
    daq.BooleanSwitch(id='toggle-general-outlier', on=False, style=
                        {'display':'inline-block', 
                        'width':'100px', 
                        'margin-bottom':'5px',
                        'float':'right'
                        }),
    dcc.Slider(0.01, 0.40, 0.01,
                value=0.05,
                marks= {
                0.01: '1%',
                0.05: '5%',
                0.10: '10%',
                0.15: '15%',
                0.20: '20%',
                0.25: '25%',
                0.30: '30%',
                0.35: '35%',
                0.40: '40%',
                },
                id='general-outlier-threshold',
                ),
    ],
    id='general-outlier-options-div',
    style={
        'margin-top': COMPONENT_TOP_MARGIN,
        'margin-bottom': COMPONENT_BOTTOM_MARGIN,
    }
)
    
feature_bagging = html.Div([
    html.B('Feature bagging'),
    html.I(
        className="bi bi-info-circle-fill",
        id='feature-bagging-info-icon',
        style={
            'margin-left':'5px',
            'margin-right':'15px'
        }
    ),
    dbc.Tooltip(dcc.Markdown(FEATURE_BAGGING_TEXT),
        target="feature-bagging-info-icon",
    ),
    daq.BooleanSwitch(id='toggle-feature-bagging', on=False, style=
                        {'display':'inline-block', 
                        'width':'100px', 
                        'margin-bottom':'5px',
                        'float':'right',
                        }),
    dcc.Slider(0.1, 1, 0.1,
                value=0.5,
                id='feature-bagging-threshold',
                ),
    ],
    id='feature-bagging-options-div',
    style= {
        'margin-bottom':COMPONENT_BOTTOM_MARGIN,
        'margin-top': COMPONENT_TOP_MARGIN,
    },
    hidden=True,
)

random_forest = html.Div([
    html.B('Random forest'),
    html.I(
        className="bi bi-info-circle-fill",
        id='random-forest-info-icon',
        style={
            'margin-left':'5px',
            'margin-right':'15px'
        }
    ),
    dbc.Tooltip(dcc.Markdown(RANDOM_FOREST_TEXT),
        target="random-forest-info-icon",
    ),
    daq.BooleanSwitch(id='toggle-random-forest', on=False, style=
                        {'display':'inline-block', 
                        'width':'100px', 
                        'margin-bottom':'5px',
                        'float':'right',
                        }),
    dcc.Slider(0.1, 0.9, 0.1,
                value=0.5,
                id='random-forest-threshold',
                ),
    ],
    id='random-forest-options-div',
    style={
        'margin-bottom':COMPONENT_BOTTOM_MARGIN,
        'margin-top': COMPONENT_TOP_MARGIN,
        },
    hidden=True,
)

    # Neural training (unique values)

neural_values = html.Div([
    html.B('Training predictions'),
    html.I(
        className="bi bi-info-circle-fill",
        id='neural-training-values-info-icon',
        style={
            'margin-left':'5px',
            'margin-right':'15px'
        }
    ),
    dbc.Tooltip(dcc.Markdown(NEURAL_VALUES_TEXT),        
        target="neural-training-values-info-icon",
    ),
    daq.BooleanSwitch(id='toggle-neural-training-values', on=False, style=
                        {'display':'inline-block', 
                        'width':'100px', 
                        'margin-bottom':'5px',
                        'float':'right'
                        }),
    dcc.Slider(0.2, 1, 0.1,
                value=0.5,
                id='neural-training-values-threshold',
                marks= {
                0.2: '2',
                0.3: '3',
                0.4: '4',
                0.5: '5',
                0.6: '6',
                0.7: '7',
                0.8: '8',
                0.9: '9',
                1: '10',
                },
                ),
    ],
    id='neural-training-values-options-div',
    style={
        'margin-bottom':COMPONENT_BOTTOM_MARGIN,
        'margin-top': COMPONENT_TOP_MARGIN,
        },
    hidden=True,
)

    # Neural training (switches)
    
neural_switches = html.Div([
    html.B('Training switches'),
    html.I(
        className="bi bi-info-circle-fill",
        id='neural-training-switches-info-icon',
        style={
            'margin-left':'5px',
            'margin-right':'15px'
        }
    ),
    dbc.Tooltip(dcc.Markdown(NEURAL_SWITCHES_TEXT),
        target="neural-training-switches-info-icon",
    ),
    daq.BooleanSwitch(id='toggle-neural-training-switches', on=False, style=
                        {'display':'inline-block', 
                        'width':'100px', 
                        'margin-bottom':'5px',
                        'float':'right'
                        }),
    dcc.Slider(0.1, 1, 0.1,
                value=0.5,
                id='neural-training-switches-threshold',
                marks= {
                0.1: '1',
                0.2: '2',
                0.3: '3',
                0.4: '4',
                0.5: '5',
                0.6: '6',
                0.7: '7',
                0.8: '8',
                0.9: '9',
                },
                ),
    ],
    id='neural-training-switches-options-div',
    style={
        'margin-bottom': COMPONENT_BOTTOM_MARGIN,
        'margin-top': COMPONENT_TOP_MARGIN,
        },
    hidden=True,
)

cleanlab = html.Div([
    html.B('CleanLab'),
    html.I(
        className="bi bi-info-circle-fill",
        id='cleanlab-info-icon',
        style={
            'margin-left':'5px',
            'margin-right':'15px'
        }
    ),
    dbc.Tooltip(dcc.Markdown(CLEANLAB_TEXT),
        target="cleanlab-info-icon",
    ),
    daq.BooleanSwitch(id='toggle-cleanlab', on=False, style=
                        {'display':'inline-block', 
                        'width':'100px',
                        'float':'right'
                        })
    ],
    id='cleanlab-options-div',
    style={
        'margin-bottom': COMPONENT_BOTTOM_MARGIN,
        'margin-top': COMPONENT_TOP_MARGIN,
        },
    hidden=True,
)

    # Outlier intersection vs union
    
intersection_toggle = html.Div([
    html.P('Intersection', style={'display':'inline-block'}),
    daq.ToggleSwitch(
        id='toggle-union',
        value=False,
        style={'display':'inline-block', 'margin-left':'5px', 'margin-right':'5px'}
    ),
    html.P('Union', style={'display':'inline-block'}),
    ],
    id='union-div',
    style={
        'margin-top': COMPONENT_TOP_MARGIN,
        'margin-bottom': COMPONENT_BOTTOM_MARGIN,
    },
)

label_error = html.Div([
    html.B('Label error detection'),
    html.I(
        className="bi bi-info-circle-fill",
        id='label-error-info-icon',
        style={
            'margin-left':'5px',
            'margin-right':'15px'
        }
    ),
    dbc.Tooltip(dcc.Markdown(LABEL_ERROR_TEXT),
        target="label-error-info-icon",
    ),
    daq.BooleanSwitch(
        id='toggle-label-errors',
        on=False,
        style={'display':'inline-block', 
                'width':'100px', 
                'margin-bottom':'5px',
                'float':'right'
            }
    ),
    dcc.Slider(0.5, 1, 0.1,
                value=0.8,
                id='label-error-threshold',
                ),
    ],
    id='toggle-label-errors-div',
    style={
        'margin-bottom':COMPONENT_BOTTOM_MARGIN,
        'margin-top':COMPONENT_TOP_MARGIN
        },
    hidden=True,
)