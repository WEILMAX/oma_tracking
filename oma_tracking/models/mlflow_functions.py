import subprocess
import webbrowser
import warnings
import mlflow

def connect_mlflow_ui(
        mlflow_ui_string: str,
        database_url: str = 'http://127.0.0.1:5000',
        ) -> None:
    """Connect to mlflow UI through the mlflow ui string and the database url
    
    Args:
        mlflow_ui_string (str): The mlflow ui string.
        database_url (str, optional): The database url.
            Defaults to 'http://127.0.0.1:5000'

    """
    process = subprocess.Popen(mlflow_ui_string, shell=True)
    webbrowser.open(database_url)
    process.terminate()


def create_mlflow_tracking_uri(
    psql_user: str = "psql_username",
    psql_password: str = "psql_password",
    psql_ip_adress: str = "10.0.0.139",
    psql_database: str = "mlflow_db",
    port_nr: str = "5432",
) -> str:
    """Create the mlflow tracking uri from the username, password, ip adress and database.
    The mlflow tracking uri is in the shape of:
        'postgresql+psycopg2://<psql_username>:"<psql_password>"@<psql_ip_adress>(:port_nr)/<mlflow_db>'

    Args:
        psql_user (str, optional): Username to acces PostgreSQL databse.
            Defaults to "psql_username".
        psql_password (str, optional): Password to acces PostgreSQL databse.
            Defaults to "psql_password".
        psql_ip_adress (str, optional): IP adress of the PostgreSQL database.
            Defaults to "10.0.0.139".
        psql_database (str, optional): Name of the PostgreSQL database.
            Defaults to "mlflow_db".
        port_nr (str, optional): Port number of the PostgreSQL database, not required for access.
            Defaults to "5432".

    Returns:
        str: mlflow_tracking_uri
    """
    mlflow_tracking_uri = (
        "postgresql+psycopg2://"
        + psql_user
        + ":"
        + psql_password
        + "@"
        + psql_ip_adress
    )
    if port_nr != "":
        mlflow_tracking_uri += ":" + port_nr
    mlflow_tracking_uri += "/" + psql_database
    return mlflow_tracking_uri


def create_mlflow_ui(
    mlflow_tracking_uri: str = "",
    artifact_root: str = "wasbs://test@mlflowstoragev1.blob.core.windows.net",
    psql_user: str = "",
    psql_password: str = "",
    psql_ip_adress: str = "10.0.0.139",
    psql_database: str = "mlflow_db",
    port_nr: str = "5432",
) -> str:
    """Connect to the mlflow ui by running the mlflow_ui_string as command line in a terminal.

    Args:
        mlflow_tracking_uri (str): Environmental variable for mlflow tracking
        artifact_root (str): Connection string to blob storage, where models will be saved as artifacts.
                             Defaults to 'wasbs://test@mlflowstoragev1.blob.core.windows.net'.
                             more information at:
                             https://www.mlflow.org/docs/latest/tracking.html#backend-stores
        psql_user (str, optional): Username to acces PostgreSQL databse.
            Defaults to "psql_username".
        psql_password (str, optional): Password to acces PostgreSQL databse.
            Defaults to "psql_password".
        psql_ip_adress (str, optional): IP adress of the PostgreSQL database.
            Defaults to "10.0.0.139".
        psql_database (str, optional): Name of the PostgreSQL database.
            Defaults to "mlflow_db".
        port_nr (str, optional): Port number of the PostgreSQL database, not required for access.
            Defaults to "5432".

    Raises:
        ValueError: psql_user can't have colons (:).
        ValueError: psql_user can't have colons @.
        ValueError: The mlflow_tracking_uri needs to have at least two colons (:).
        ValueError: The mlflow_tracking_uri needs to have at least one @.
        If an error is raised, run mlflow ui manually!

    Returns:
        str: mlflow ui string to run in a terminal
    """
    if mlflow_tracking_uri == "":
        mlflow_tracking_uri = create_mlflow_tracking_uri(
            psql_user=psql_user,
            psql_password=psql_password,
            psql_ip_adress=psql_ip_adress,
            psql_database=psql_database,
            port_nr=port_nr,
        )

        if psql_user.count(":") > 0:
            raise ValueError(
                "The psql username in the mlflow_tracking_uri has colons (:), it has "
                + str(mlflow_tracking_uri.count(":"))
                + ". Check mlflow_tracking_uri or run mlflow ui manually"
            )
        if psql_user.count("@") > 0:
            raise ValueError(
                "The psql username in the mlflow_tracking_uri has @ symbols, it has "
                + str(mlflow_tracking_uri.count("@")),
                ". Check mlflow_tracking_uri or run mlflow ui manually",
            )

    else:
        warnings.warn(
            "Mlflow_tracking_uri passed without checking checking username for ':' and '@' symbols."
            + " Manually control the uri!"
        )

    if mlflow_tracking_uri.count(":") < 2:
        raise ValueError(
            "The mlflow_tracking_uri doesn't have at least 2 colons (:), it has "
            + str(mlflow_tracking_uri.count(":"))
            + ". Check mlflow_tracking_uri or run mlflow ui manually"
        )
    if mlflow_tracking_uri.count("@") < 1:
        raise ValueError(
            "The mlflow_tracking_uri doesn't have 1 @ symbol, it has "
            + str(mlflow_tracking_uri.count("@")),
            ". Check mlflow_tracking_uri or run mlflow ui manually",
        )

    l1 = "@".join(mlflow_tracking_uri.split("@")[0:-1])
    l2 = mlflow_tracking_uri.split("@")[-1]

    mlflow_ui_string = (
        "mlflow ui --backend-store-uri "
        + l1.split(":", maxsplit=1)[0]  # 'postgresql+psycopg2'
        + ":"
        + l1.split(":")[1]  # '//psql_user'
        + ':\"'
        + ":".join(l1.split(":")[2:])  # 'psql_password'
        + '\"@'
        + l2  # ip_number(:gate_number)/database
        + " --default-artifact-root "
        + artifact_root
    )
    return mlflow_ui_string


def run_mlflow_experiment(
    project: str = '',
    artifact_root="wasbs://test@mlflowstoragev1.blob.core.windows.net",
    location="no_location",
    experiment_nr=1,
    experiment_name: str = '',
):
    """Set the mlflow experiment and create it in the artifact_root if it doesn't esxist yet.
    The experiment name is set to experiment_name variable if this is given,
    otherwise it becomes <project>_<location>_experiment_<experiment_nr>.

    Args:
        project (str, optional): Name of the project.
            Defaults to ''.
        artifact_root (str, optional): Connection string to blob storage,
            where models will be saved as artifacts.
            Defaults to 'wasbs://test@mlflowstoragev1.blob.core.windows.net'.
            more information at:
            https://www.mlflow.org/docs/latest/tracking.html#backend-stores
        location (str, optional): Location of the analysed structure. Defaults to "no_location".
        experiment_nr (int, optional): Index of the experiment run. Defaults to 1.
        experiment_name (str, optional): Experiment name if it deviates 
            from the automatically generated one.
            Defaults to ''.

    Returns:
        str: Name of the currently active experiment name.
    """
    if experiment_name == '':
        experiment_name = "_".join(
            [str(project), str(location), "experiment", str(experiment_nr)]
        )
    if mlflow.get_experiment_by_name(experiment_name) is None:
        print("created mlflow experiment: " + experiment_name)
        mlflow.create_experiment(experiment_name, artifact_root)
    mlflow.set_experiment(experiment_name)
    print("mlflow experiment set to: " + experiment_name)
    return experiment_name


def load_specific_dw_model(
    model_name: str, model_version: int = None, model_type: str = None
):
    """Load ML model through mlflow based on the model name and model version.
    More information of the model can be found in the mlflow ui.

    Args:
        model_name (str): The model name, follows the following construction:
            <project>_<location>_<model_function>(_<class_regr_parameter>)
        model_version (int, optional): Version of the model that is loaded.. Defaults to None.
        model_type (str, optional): Model of type keras, sklearn or other.

    Returns:
        mlflow.keras or mlflow.sklearn or mlflow.pyfunc Model depending on the model type specified.
    """
    model_uri = f"models:/{model_name}/{model_version}"

    if model_type and "keras" in model_type:
        model = mlflow.keras.load_model(model_uri)
    if model_type and "sklearn" in model_type:
        model = mlflow.sklearn.load_model(model_uri)
    if model_type and "xgboost" in model_type:
        model = mlflow.sklearn.load_model(model_uri)
    else:
        model = mlflow.pyfunc.load_model(model_uri)
    return model
