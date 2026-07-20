"""
run_experiments.py

Main entry point for forecasting experiments.

Supports:
- Company Forecasting
- Store Forecasting
- Store + Department Forecasting
"""

from pipeline.training.champion_pipeline import (
    ChampionPipeline
)

from pipeline.preprocessing.data_loader import (
    WalmartDataLoader
)

from pipeline.preprocessing.aggregations import (
    WalmartAggregator
)

from pipeline.forecasting.sarima import (
    SARIMAForecaster
)

from pipeline.forecasting.prophet import (
    ProphetForecaster
)

from pipeline.forecasting.xgboost import (
    XGBoostForecaster
)

from pipeline.training.experiment_runner import (
    ExperimentRunner
)


def create_models():
    """
    Create forecasting models.
    """

    return [

        SARIMAForecaster(
            seasonal_period=12
        ),

        ProphetForecaster(),

        XGBoostForecaster(
            n_estimators=50
        )

    ]


def run_company_forecast(
    train_df
):
    """
    Company level forecasting.
    """

    aggregator = WalmartAggregator()

    company_df = (
        aggregator.get_company_sales(
            train_df
        )
    )

    runner = ExperimentRunner(
        models=create_models()
    )

    leaderboard = runner.run(
        company_df
    )

    champion = (
        runner.get_champion()
    )

    champion_pipeline = ChampionPipeline()

    pipeline_result = (
        champion_pipeline.run(
            champion_result=champion,
            full_dataset=company_df
        )
    )

    champion_pipeline.print_results(
        pipeline_result
    )



    print("\n")
    print("=" * 60)
    print("COMPANY FORECASTING")
    print("=" * 60)

    print(leaderboard)

    print("\nChampion:")
    print(
        champion["model_name"]
    )


def run_store_forecast(
    train_df,
    store_id=1
):
    """
    Store level forecasting.
    """

    aggregator = WalmartAggregator()

    store_df = (
        aggregator.get_store_sales(
            train_df,
            store_id=store_id
        )
    )

    runner = ExperimentRunner(
        models=create_models()
    )

    leaderboard = runner.run(
        store_df
    )

    champion = (
        runner.get_champion()
    )

    champion_pipeline = ChampionPipeline()

    pipeline_result = (
        champion_pipeline.run(
            champion_result=champion,
            full_dataset=store_df
        )
    )

    champion_pipeline.print_results(
        pipeline_result
    )

    print("\n")
    print("=" * 60)
    print(
        f"STORE {store_id} FORECASTING"
    )
    print("=" * 60)

    print(leaderboard)

    print("\nChampion:")
    print(
        champion["model_name"]
    )


def run_store_department_forecast(
    train_df,
    store_id=1,
    dept_id=1
):
    """
    Store + Department forecasting.
    """

    aggregator = WalmartAggregator()

    store_dept_df = (
        aggregator
        .get_store_department_sales(
            train_df,
            store_id=store_id,
            dept_id=dept_id
        )
    )

    runner = ExperimentRunner(
        models=create_models()
    )

    leaderboard = runner.run(
        store_dept_df
    )

    champion = (
        runner.get_champion()
    )



    champion_pipeline = ChampionPipeline()

    pipeline_result = (
        champion_pipeline.run(
            champion_result=champion,
            full_dataset=store_dept_df
        )
    )

    champion_pipeline.print_results(
        pipeline_result
    )


    # from pipeline.training.model_logger import (
    # ModelLogger
    # )

    # model_logger = ModelLogger()

    # model_uri = model_logger.log_model(
    #     model_name=champion["model_name"],
    #     trained_model=champion["model_object"]
    # )

    # print(
    #     f"Champion Model URI: {model_uri}"
    # )

    print("\n")
    print("=" * 60)
    print(
        f"STORE {store_id}"
        f" DEPT {dept_id}"
    )
    print("=" * 60)

    print(leaderboard)

    print("\nChampion:")
    print(
        champion["model_name"]
    )


def main():

    data_loader = (
        WalmartDataLoader()
    )

    train_df = (
        data_loader.load_train_data()
    )

    run_company_forecast(
        train_df
    )

    # Uncomment as needed

    # run_store_forecast(
    #     train_df,
    #     store_id=1
    # )

    # run_store_department_forecast(
    #     train_df,
    #     store_id=1,
    #     dept_id=1
    # )


if __name__ == "__main__":
    main()
