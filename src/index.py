import base64
import io
import os
import webbrowser

import dash_bootstrap_components as dbc
import matplotlib
import pandas as pd

from app import app
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from explainerdashboard import ClassifierExplainer, ExplainerDashboard, ExplainerHub
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from tabs.AutoMLReportTab import AutoMLReportTab
from tabs.FeaturesImportancesTab import FeaturesImportanceBasicTab, FeaturesImportanceExpertTab
from tabs.ClassificationStatsTab import ClassificationStatsTab
from tabs.WhatIfTab import WhatIfBasicTab, WhatIfExpertTab
from tabs.CounterfactualsTab import CounterfactualsTab
from waitress import serve

# Select agg as matplotlib backend.
matplotlib.use('agg')

# Alert for when an invalid file is uploaded.
invalid_type_alert = dbc.Alert(
    children="Invalid dataset type. Please be sure it is a .csv or .xlsx file.", 
    color="danger",
    dismissable=True,
    duration=10000
),

# Alert for when Start button is pressed without a file uploaded.
no_file_alert = dbc.Alert(
        children="Please upload the dataset before pressing Start.", 
        color="danger",
        dismissable=True,
        duration=10000
),

# Layout of the home page.
app.layout = dbc.Container([
    html.Div([          
        #NAVBAR
        dbc.Navbar(
            children=[
                html.A(
                    dbc.Row([
                        dbc.Col(
                            # Creating a brand for the navbar.
                            dbc.NavbarBrand(
                                "Democratizing the prediction of student's dropout: The Interactive Visual DashBoard tool", className="ms-2"
                            )
                        ),
                        ],                                                                     
                        className="g-0 ml-auto flex-nowrap mt-3 mt-md-0",
                        align="center",
                    ),
                ),
            ],     
        ),

        #BODY
        html.Div(id='upload-div',
            children=[
                html.Br(),
                dbc.CardBody([
                    dcc.Markdown(
                        """
                        Welcome!

                        Before you upload a file, please make sure that it meets the following requirements:
                         - Check that you are uploading a .csv or .xlsx file.
                         - Make sure that the first column of the dataset is reserved for the index.
                         - At least one of the columns should contain continuous values (represented as floating-point numbers)
                         - The last column should have only two unique values: "Dropout" and "No dropout".

                         The code of this app is available at: https://github.com/0Kan0/Democratizing-the-prediction-of-student-s-dropout-The-Interactive-Visual-DashBoard-tool
                        """,
                        style={"margin": "0 10px"},
                    )
                ]),

                html.Br(),
                html.H3("Load dataset"),

                # A component that allows to upload a file.
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                    multiple=False,
                ),       
            ]),

        html.Br(),
        dbc.Spinner(dbc.Row(
            dbc.Col(
                [
                    # Creating a table with the data.
                    html.Div(id="output-data"),
                ]
            ),
        )),

        html.Br(),
        # Button to start the whole process of training the machine learning model and building the dashboard.
        dbc.Button(f"Start", color="primary", id="start-button", style={'textAlign': 'center'}),

        html.Br(),
        dbc.Spinner(
            html.Div(
                id='dashboard-button',
                # Invisible button that, when the whole proccess finish, appears and redirect to the dashboard.
                children=dbc.Button(
                    f"Go to dashboard",
                    id='dashboard-button-link',
                    href="http://127.0.0.1:8050/", 
                    target="_blank",
                ),
                hidden=True
            ),
        ),
        
        html.Br(),
        # Component to display the button that redirects to the dashboard.
        html.Div(id='placeholder'),

        html.Br(),
        # Component that shows alerts.
        html.Div(id="alert", children=[]),
    ])
])

def parse_data(contents, filename): 
    """
    Parse data function to convert uploaded file content to pandas dataframe.

    Args:
        - contents (str): The content of the uploaded file.
        - filename (str): The name of the uploaded file.

    Returns:
        - A pandas.DataFrame containing the data from the uploaded file.
    """

    if contents:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        if "csv" in filename:
            # Assume that the user uploaded a CSV
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=',', decimal='.')
        elif "xlsx" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

        # Return the pandas.DataFrame of the uploaded file
        return df
    
def create_AutoML_model(contents, filename):
    """
    Create an AutoML model using the provided dataset.

    Args:
        - contents (str): The contents of the dataset in CSV format.
        - filename (str): The filename of the dataset.

    Returns:
        - tuple: A tuple containing the following items:
            - The trained `model` object
            - The `X_test` and `y_test` datasets for evaluating the performance of the trained model
            - A `model_report` which contains information about the performance of each algorithm tried by the AutoML model
            - The `trained_model` object which contains the trained model
            - The original dataset `df`.
            - Bool that indicates if the model was loaded or was trained.
    """

    # Parse the CSV data and set the first column as the index
    df = parse_data(contents, filename)
    df.set_index(df.columns[0], inplace=True)

    # Replace class labels with binary values (0 or 1)
    df.iloc[:, -1] = df.iloc[:, -1].replace({"Dropout": 0, "No dropout": 1})

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        df[df.columns[:-1]], df.iloc[:, -1], test_size=0.20
    )

    # Set saved models folder path
    saved_models_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "saved_AutoML_models", filename.split(".")[0])

    # If dataset was already loaded, don't need to create a new model
    if os.path.exists(saved_models_path):
        model = AutoML(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "saved_AutoML_models", filename.split(".")[0]))

        model_report = model.report()

        return model, X_test, y_test, model_report, model, df, True


    # Define the AutoML model configuration
    model = AutoML(
            results_path=saved_models_path,
            algorithms=["Baseline", "Linear", "Decision Tree", "Random Forest", "Extra Trees", "Xgboost", "LightGBM", "CatBoost", "Neural Network", "Nearest Neighbors"],
            start_random_models=1,
            stack_models=True,
            train_ensemble=True,
            explain_level=2,
            validation_strategy={
                "validation_type": "split",
                "train_ratio": 0.80,
                "shuffle" : True,
                "stratify" : True,
            })
    
    # Train the AutoML model
    trained_model = model.fit(X_train, y_train)

    # Get the performance report for the AutoML model
    model_report = model.report()

    # Return the trained model object along with other useful objects for evaluating the model's performance
    return model, X_test, y_test, model_report, trained_model, df, False

@app.callback(
        Output('output-data','children'),
        [Input('upload-data', 'contents'),
        Input('upload-data', 'filename')],
        prevent_initial_call=True
)
def update_table(contents, filename):
    """
    Update the table with the parsed data and create a Dash DataTable object that allows the user to view the data.

    Args:
        - contents (str): The contents of the file that is uploaded by the user.
        - filename (str): The name of the file that is uploaded by the user.

    Returns:
        - A Dash Div object containing a DataTable that displays the parsed data, the name of the uploaded file, 
        and the raw content of the file. Or an alert if the file uploaded was not a .csv or .xlsx file.
    """

    if contents:
        try:
            df=parse_data(contents,filename)
        except Exception:
            return invalid_type_alert
        
        table = html.Div(
            [
                html.H4(filename),
                dash_table.DataTable(
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict("records"),
                    style_table={'overflowX': 'scroll'},
                    sort_mode='multi',
                    row_deletable=False,
                    column_selectable='single',
                    selected_columns=[],
                    selected_rows=[],
                    page_action='native',
                    page_current= 0,
                    page_size= 20,
                    style_data_conditional=[        
                        {'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'}
                    ],
                ),
                html.Hr(),
                html.Div("Raw Content"),
                html.Pre(
                    contents[0:200] + "...",
                    style={"whiteSpace": "pre-wrap", "wordBreak": "break-all"},
                ),
            ]
        )
        return table
    
@app.callback(
        Output('alert', 'children'),
        Output('dashboard-button', 'hidden'),
        [State('upload-data', 'contents'),
         State('upload-data', 'filename')],
        Input('start-button', 'n_clicks'),
        prevent_initial_call=True
)
def create_dashboard(contents, filename, n_clicks):
    """
    Create a dashboard using the contents of a file and return a boolean and an alert message.

    Args:
        - contents (str): The contents of the file.
        - filename (str): The name of the file.
        - n_clicks (int or None): The number of clicks.

    Returns:
        - tuple: A tuple containing the following items:
            - A boolean changing the property "hidden" of the 'dashboard-button' html.Div.
            - Alert message if no file was uploaded and Start button was clicked.
    """

    # Return an alert if any of the parameters are None and mantain the html.Div hidden
    if (contents is None or filename is None or n_clicks is None):
        return no_file_alert, True

    # Global variable for the hub
    global hub

    # Create an AutoML model using the contents of the file
    model, X_test, y_test, reportML, trained, df, loaded = create_AutoML_model(contents, filename)

    # If model dataset was already loaded, load explainer
    if loaded:

        # Set saved models folder path
        saved_explainer_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "saved_AutoML_models", filename.split(".")[0], "explainer.dill")

        # Create a classifier explainer object
        explainer = ClassifierExplainer.from_file(saved_explainer_path)


        # Create two explainer dashboards with different tabs
        db1 = ExplainerDashboard(explainer, header_hide_selector=True, hide_poweredby=True, title="AutoML Student Dropout Explainer (Basic Interface)", 
                                tabs=[FeaturesImportanceBasicTab, WhatIfBasicTab],
                                description="In this dashboard, you can access the following tabs: Prediction and What If...")
        
        db2 = ExplainerDashboard(explainer, header_hide_selector=True, hide_poweredby=True, title="AutoML Student Dropout Explainer (Advanced Interface)",
                                tabs=[AutoMLReportTab(explainer=explainer, ML_report=reportML), FeaturesImportanceExpertTab, ClassificationStatsTab, WhatIfExpertTab, CounterfactualsTab(explainer=explainer, dataframe=df, trained_model=trained)],
                                description="In this dashboard, you can access the following tabs: AutoML Report, Feature Importances, Classificaction Stats, What If... and Counterfactual Scenarios.")
        
        # Create an explainer hub with the two dashboards
        hub = ExplainerHub([db1, db2], title="Democratizing the prediction of student's dropout: The Interactive Visual DashBoard tool", n_dashboard_cols=2,
                            description="")

        # Return no alert message and reveals the html.Div
        return None, False
    
    # Create a classifier explainer object
    explainer = ClassifierExplainer(model, X_test, y_test, labels=["Dropout", "No dropout"], target="Target")

    # Create two explainer dashboards with different tabs
    db1 = ExplainerDashboard(explainer, header_hide_selector=True, hide_poweredby=True, title="AutoML Student Dropout Explainer (Basic Interface)", 
                            tabs=[FeaturesImportanceBasicTab, WhatIfBasicTab],
                            description="In this dashboard, you can access the following tabs: Prediction and What If...")
    
    db2 = ExplainerDashboard(explainer, header_hide_selector=True, hide_poweredby=True, title="AutoML Student Dropout Explainer (Advanced Interface)",
                            tabs=[AutoMLReportTab(explainer=explainer, ML_report=reportML), FeaturesImportanceExpertTab, ClassificationStatsTab, WhatIfExpertTab, CounterfactualsTab(explainer=explainer, dataframe=df, trained_model=trained)],
                            description="In this dashboard, you can access the following tabs: AutoML Report, Feature Importances, Classificaction Stats, What If... and Counterfactual Scenarios.")
    
    # Create an explainer hub with the two dashboards
    hub = ExplainerHub([db1, db2], title="Democratizing the prediction of student's dropout: The Interactive Visual DashBoard tool", n_dashboard_cols=2,
                        description="")
    
    # Set saved models folder path
    saved_explainer_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "saved_AutoML_models", filename.split(".")[0], "explainer.dill")

    # Save explainer in the models path
    explainer.dump(saved_explainer_path)

    # Return no alert message and reveals the html.Div
    return None, False

@app.callback(
        Output('placeholder', 'children'),
        Input('dashboard-button-link', 'n_clicks'),
        prevent_initial_call=True
)
def start_dashboard(n_clicks):
    """
    Starts the dashboard by running the ExplainerHub.

    Args:
        - n_clicks (int or None): The number of times the start button has been clicked.

    Returns:
        - None if n_clicks is None.
        - The output of the hub.run() method (an HTML object) if n_clicks is not None.
    """
    
    if(n_clicks is None):
        return None
    
    return hub.run(use_waitress=True)

# A way to run the app in a local server.
if __name__ == '__main__':
    app.title = "Democratizing the prediction of student's dropout: The Interactive Visual DashBoard tool"
    
    webbrowser.open_new_tab('http://localhost:8080/')
    serve(app.server, host='0.0.0.0', port=8080)
    """ app.run(debug=False) """


