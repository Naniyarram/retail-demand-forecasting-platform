"""
API entry point for the forecasting service application.
"""

import uvicorn


def main():
    """
    Starts the FastAPI server locally using uvicorn.
    """

    print(
        "Starting Retail Demand Forecasting API at "
        "http://127.0.0.1:8000"
    )
    print(
        "Press Ctrl+C to stop the server."
    )

    uvicorn.run(
        "pipeline.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    main()
