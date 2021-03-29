def deploy_model_from_databricks(
    mlflow, 
    splice, 
    model_flavor, 
    db_schema_name,
    db_table_name,
    run_id,
    reference_table=None,
    reference_schema=None,
    primary_key=None,
    df=None,
    create_model_table=True,
    model_cols=None,
    classes=None,
    library_specific=None,
    replace=False,
) -> None:
    from tempfile import TemporaryDirectory
    from io import BytesIO
    from zipfile import ZIP_DEFLATED, ZipFile
    import re
    # Get the model using their mlflow before getting our own
    print(f'Getting databricks {model_flavor} model')
    model = getattr(mlflow, model_flavor).load_model(run_id)
    try:
        # Override mlflow
        from splicemachine.mlflow_support import mlflow_support
        mlflow_support.main()
        mlflow = mlflow_support.mlflow
        from splicemachine.mlflow_support.constants import FileExtensions
        from splicemachine import SpliceMachineException
        from splicemachine.mlflow_support.utilities import insert_artifact
        print('Automatically authenticating to Splice Machine')
        mlflow_uri = re.findall(r'jdbc-.[^:]*',splice.jdbcurl)[0].lstrip('jdbc-') + '/mlflow'
        user = re.findall(r'user=[^;]*', splice.jdbcurl)[0]
        pwd = re.findall(r'password=[^;]*', splice.jdbcurl)[0]
        mlflow.set_mlflow_uri(mlflow_uri)
        mlflow.login_director(user,pwd)
        mlflow.register_splice_context(splice)
        mlflow.set_experiment('databricks_deploy')
        if model_flavor not in FileExtensions.get_valid():
            raise SpliceMachineExceptions(f'Model flavor provided is not supported. Supported flavors are {FileExtensions.get_valid()}')
        print(f'Migrating {model_flavor} model to Splice Machine')
        mlflow.start_run(run_name=f'Databricks run {run_id}')
        mlflow.log_model(model, model_lib=model_flavor)
        print(f'Deploying {model_flavor} Model')
        jid = mlflow.deploy_database(db_schema_name, db_table_name, mlflow.current_run_id(), reference_table, reference_schema, primary_key,
                              df, create_model_table, model_cols, classes, library_specific, replace, False)
        mlflow.watch_job(jid)
        mlflow.end_run()
    finally:
        import importlib, mlflow # Try to get back the databricks mlflow
        importlib.reload(mlflow)

