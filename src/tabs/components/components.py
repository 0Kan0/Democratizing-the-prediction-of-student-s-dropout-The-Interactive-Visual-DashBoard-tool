from io import StringIO

import dash_bootstrap_components as dbc
import dice_ml as dml
import pandas as pd

import time
from dash import dcc, html, Input, Output, State, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from explainerdashboard.custom import *

class AutoMLReportComponent(ExplainerComponent):
    """
    A component to display the AutoML report in the dashboard. Inherits from ExplainerComponent.
    """
    
    def __init__(self, explainer, title="AutoML Report", name=None, 
                        subtitle="Compare all models used and see which one fits the best", 
                        ML_report=None, **kwargs):
        """
        Initialize the AutoMLReportComponent.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): The name of the component.
            - subtitle (str): The subtitle of the component.
            - ML_report (html.Iframe): The AutoML report.
            - **kwargs: Optional keyword arguments.
        """
        
        # Call the parent constructor
        super().__init__(explainer, title, name, **kwargs)

        # Set the ML_report attribute
        self.ML_report = ML_report
    
    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap card
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    # Add the title and subtitle
                    html.H3(self.title, className="card-title", id='automlreport-title-'+self.name),
                    html.H6(self.subtitle, className='card-subtitle')
                ]),
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        # Add an Iframe to display the AutoML report
                        html.Iframe(srcDoc=self.ML_report.data, style={'width': '100%', 'height': '1000px'})
                    ]),
                ], class_name="mt-4")
            ])
        ], class_name="h-100")

class SelectStudentComponent(ExplainerComponent):
    """
    Component for selecting a student for a classification model.
    """

    def __init__(self, explainer, title="Select Random Student", name=None,
                        subtitle="Select from list or pick at random",
                        index_dropdown=True,
                        pos_label=None, index=None, slider= None, labels=None,
                        pred_or_perc='predictions', description=None,
                        **kwargs):
        """
        Initialize SelectStudentComponent.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component
            - name (str): The name of the component.
            - subtitle (str): The subtitle of the component.
            - index_dropdown (bool): If True, a dropdown will be used for selecting the student index. Otherwise, an input field will be displayed.
            - pos_label (any): Positive class label to select.
            - index (str): Default index value.
            - slider (list of float or None): Slider range for selecting predicted probabilities
            - labels (list of str or None): Class labels to select
            - pred_or_perc (str): Whether to display predictions or percentiles
            - description (str or None): Description of the component
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Set name for index dropdown
        self.index_name = 'random-index-clas-index-'+self.name

        # Set slider range for selecting predicted probabilities
        if self.slider is None:
            self.slider = [0.0, 1.0]

        # Set class labels to select
        if self.labels is None:
            self.labels = self.explainer.labels

        # Hide labels if y is missing
        if self.explainer.y_missing:
            self.hide_labels = True

        # Initialize PosLabelSelector and IndexSelector
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_selector = IndexSelector(explainer, 'random-index-clas-index-'+self.name,
                                    index=index, index_dropdown=True, **kwargs)

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap card
        return dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        # Add the title and subtitle
                        html.H3(f"Select student", id='random-index-clas-title-'+self.name),
                        html.H6(self.subtitle, className='card-subtitle'), 
                    ]), 
                ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        self.index_selector.layout()
                    ], width=8, md=8), 
                    make_hideable(
                        dbc.Col([
                            self.selector.layout()
                        ], md=2), 
                    hide=True),
                ], class_name="mb-2"),

                dbc.Row([
                    dbc.Col([
                        dbc.Button(f"Random student", color="primary", id='random-index-clas-button-'+self.name),
                    ], width=4, md=4), 
                ], class_name="mb-2"),
                
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.Label(f"Observed {self.explainer.target}:", id='random-index-clas-labels-label-'+self.name),
                            dcc.Dropdown(
                                id='random-index-clas-labels-'+self.name,
                                options=[{'label': lab, 'value': lab} for lab in self.explainer.labels],
                                multi=True,
                                value=self.labels),
                        ], width=8, md=8),
                        hide=True),
                    make_hideable(
                        dbc.Col([
                            dbc.Label(
                                "Range:", html_for='random-index-clas-pred-or-perc-'+self.name),
                            dbc.Select(
                                id='random-index-clas-pred-or-perc-'+self.name,
                                options=[
                                    {'label': 'probability',
                                        'value': 'predictions'},
                                    {'label': 'percentile',
                                        'value': 'percentiles'},
                                ],
                                value=self.pred_or_perc),
                        ], width=4,
                        id='random-index-clas-pred-or-perc-div-'+self.name),
                        hide=True)
                ], class_name="mb-2"),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            html.Div([
                                dbc.Label(id='random-index-clas-slider-label-'+self.name,
                                    children="Predicted probability range:",
                                    html_for='prediction-range-slider-'+self.name),
                                dcc.RangeSlider(
                                    id='random-index-clas-slider-'+self.name,
                                    min=0.0, max=1.0, step=0.01,
                                    value=self.slider,  allowCross=False,
                                    marks={0.0:'0.0', 0.2:'0.2', 0.4:'0.4', 0.6:'0.6', 
                                            0.8:'0.8', 1.0:'1.0'},
                                    tooltip = {'always_visible' : False})
                            ])
                        ]), 
                    hide=True),
                ], justify="start"),
            ]),
        ], class_name="h-100")

    def component_callbacks(self, app):
        @app.callback(
            Output('random-index-clas-index-'+self.name, 'value'),
            [Input('random-index-clas-button-'+self.name, 'n_clicks')],
            [State('random-index-clas-slider-'+self.name, 'value'),
             State('random-index-clas-labels-'+self.name, 'value'),
             State('pos-label-'+self.name, 'value')])
        def update_index(n_clicks, slider_range, labels, pos_label):
            if n_clicks is None and self.index is not None:
                raise PreventUpdate
            
            return self.explainer.random_index(y_values=labels,
                pred_proba_min=slider_range[0], pred_proba_max=slider_range[1],
                return_str=True, pos_label=pos_label)

class CounterfactualsComponent(ExplainerComponent):
    """
    A component for generating counterfactual scenarios for a given student in a dataset. Inherits from ExplainerComponent.
    """

    def __init__(self, explainer, title="Counterfactuals scenarios", name=None,
                        subtitle="What can a student improve?",
                        index_dropdown=True, index=None, dataframe=None, trained_model=None, **kwargs):
        """
        Initializes the CounterfactualsComponent instance.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): Name of the component (for internal use).
            - subtitle (str): Subtitle of the component.
            - index_dropdown (bool): If True, a dropdown will be used for selecting the student index. Otherwise, an input field will be displayed.
            - index (str): Default index value.
            - dataframe (Dataframe): Pandas dataframe containing the dataset.
            - trained_model (any): Trained model used for generating counterfactuals.
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Set the dataframe attribute
        self.dataframe = dataframe

        # Set the trained_model attribute
        self.trained_model = trained_model

        # Initialize IndexSelector
        self.index_selector = IndexSelector(explainer, 'random-index-clas-index-'+self.name,
                                    index=index, index_dropdown=index_dropdown, **kwargs)

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap card
        return dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        # Add the title and subtitle
                        html.H3(f"Select student and number of counterfactual scenarios", id='random-index-clas-title-'+self.name),
                        html.H6(self.subtitle, className='card-subtitle'), 
                    ]), 
                ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        # Add the index selector
                        self.index_selector.layout()
                    ], width=8, md=8), 
                ], class_name="mb-2"),

                dbc.Row([
                    dbc.Col([
                        # Add the input field for the number of scenarios to generate
                        dbc.Input(id="input",
                                  placeholder="Enter the number of scenarios between 1 and 10",
                                  type="number",
                                  debounce=True,
                                  min=1,
                                  max=10,
                                  value=None)
                    ], width=8, md=8),
                ], class_name="mb-2"),

                dbc.Row([
                    dbc.Col([
                        # Add the button to generate the scenarios
                        dbc.Button(f"Generate scenarios", color="primary", id="button"),
                    ], width=4, md=4), 
                ], class_name="mb-2"),

                html.Br(), 
                html.Br(),

                dbc.Spinner(
                    dbc.Row([
                        # Add a data table for displaying the original data
                        html.H3(f"Original", id='original-title', hidden=True),
                        dash_table.DataTable(id='original-tbl', data=None, style_table={'overflowX': 'scroll'})
                ], class_name="mb-2")),

                html.Br(),

                dbc.Spinner(
                    dbc.Row([
                        # Add a data table for displaying the generated counterfactuals
                        html.H3(f"Counterfactuals", id='counterfactuals-title', hidden=True),
                        dash_table.DataTable(id='counterfactuals-tbl', data=None, style_table={'overflowX': 'scroll'})
                ], class_name="mb-2")),
            ]),
        ], class_name="h-100")

    def component_callbacks(self, app):
        @app.callback(
            Output('counterfactuals-tbl', 'data'),
            Output('counterfactuals-tbl', 'columns'),
            Output('original-tbl', 'data'),
            Output('original-tbl', 'columns'),
            Output('counterfactuals-title', 'hidden'),
            Output('original-title', 'hidden'),
            [State('random-index-clas-index-'+self.name, 'value'),
            State('input', 'value')],
            [Input('button', 'n_clicks')]
        )

        def generate_counterfactuals(dataIndex, input, n_clicks):
            """
            Generate counterfactual scenarios given a row index and a desired number of counterfactuals.
    
            Args:
                - dataIndex (int): Index of the row in the dataframe to generate counterfactuals for.
                - input (int): Desired number of counterfactuals to generate.
                - n_clicks (int): Count of click events.
            
            Returns:
                - tuple: A tuple containing the following items: 
                    - counterfactuals_data containing the data of the counterfactuals generated.
                    - counterfactuals_columns containing the columns of the counterfactuals generated 
                    - original_data containing the data of the original dataframe. 
                    - original_columns containing the columns of the original dataframe.
                    - False that changes the visibility of both titles (Original and Counterfactuals) from hidden to visible.
            """
            
            # Check if any argument is None and return None if so
            if (dataIndex is None or input is None or n_clicks is None):
                return None

            # Get the dataframe of student data and select the row at the specified index
            student_data = self.dataframe.loc[[dataIndex]]
            student_data_deleted_target_column = student_data.drop(columns=self.dataframe.columns[-1])

            # Select continuous features from the dataframe (those that are float)
            continuous_features = student_data_deleted_target_column.select_dtypes(include=['float64']).columns.tolist()

            # Create a data object using the dataframe, continuous features and specify the outcome name
            data = dml.Data(dataframe=self.dataframe, continuous_features=continuous_features, outcome_name=self.dataframe.columns[-1])
            
            # Create a model object using the trained model and specify the backend
            model = dml.Model(model=self.trained_model, backend="sklearn")

            # Create a Dice object using the data and model objects
            exp = dml.Dice(data, model)

            # Generate counterfactuals for the selected row using the Dice object, and specify the desired number and class of counterfactuals
            dice_exp = exp.generate_counterfactuals(student_data_deleted_target_column, total_CFs=input, desired_class="opposite")
            cf_object = dice_exp.cf_examples_list[0].final_cfs_df

            # Convert the counterfactuals dataframe to a CSV string, then read it back into a pandas dataframe
            cf_object_to_csv = cf_object.to_csv(index=False)
            cf_object_to_csv = pd.read_csv(StringIO(cf_object_to_csv))

            # Replace binary values with labels (Dropout or No dropout)
            student_data.iloc[:, -1] = student_data.iloc[:, -1].replace({0: "Dropout", 1: "No dropout"})
            cf_object_to_csv.iloc[:, -1] = cf_object_to_csv.iloc[:, -1].replace({0: "Dropout", 1: "No dropout"})

            # Convert the counterfactuals dataframe and original dataframe to dictionaries of records, and create a list of column definitions
            original_data = student_data.to_dict('records')
            original_columns = [{"name": i, "id": i} for i in student_data.columns]
            counterfactuals_data = cf_object_to_csv.to_dict('records')
            counterfactuals_columns = [{"name": i, "id": i} for i in cf_object_to_csv.columns]
            
            
            # Return a tuple containing the counterfactuals data, counterfactuals columns, original data, original columns, and False values for the last two elements of the tuple
            return counterfactuals_data, counterfactuals_columns, original_data, original_columns, False, False
        
class TimerComponent(ExplainerComponent):
    def __init__(self, explainer, title="Timer", name=None, **kwargs):
        super().__init__(explainer, title, name)


    def layout(self):
        return dbc.CardBody([
                dbc.Row([
                    dcc.Store(id="time"),
                    html.H2(id="display-time", children='Start timer'),
                    html.Br(),
                    dbc.Button(f"Start", color="primary", id="start-button", style={'textAlign': 'center', 'width': '10%',}),
                    html.Br(),
                    dbc.Button(f"Stop", color="primary", id="stop-button", style={'textAlign': 'center', 'width': '10%'}),
                ], class_name="mb-2"),
        ], class_name="h-100")

    def component_callbacks(self, app):
        @app.callback(
            Output('display-time', 'children'),
            Output('time', 'data'),
            Output('stop-button', 'n_clicks'),
            Input('start-button', 'n_clicks'),
            Input('stop-button', 'n_clicks'),
            State('time', 'data'),
            prevent_initial_call=True
        )
        def update_timer(n_clicks_start, n_clicks_stop, start_time):
            if n_clicks_start and not n_clicks_stop:
                started_timer = time.time()
                return "The timer has started.", started_timer, None
            
            stopped_timer = time.time()
            try:
                elapsed_time = stopped_timer - start_time
            except:
                return 'Start timer', None, None
            
            minutes, seconds = divmod(elapsed_time, 60)
            hours, minutes = divmod(minutes, 60)

            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}", None, None
    