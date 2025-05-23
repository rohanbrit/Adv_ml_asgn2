from fastapi import FastAPI, Response
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from datetime import date, timedelta

description = """
## Rohan Rocky Britto - Student ID: 24610990

### Available APIs

You will be able to:

* **Check API Health**
* **Forecast National Sales Revenue**
* **Predict Item Sales Revenue**
"""

tags_metadata = [
    {
        "name": "health",
        "description": "This can be used to monitor the health of this API",
    },
    {
        "name": "forecast",
        "description": "Forecast the national sales revenue for the next 7 days",
    },
    {
        "name": "predict",
        "description": "Predict the sales revenue of a product in a store on a date",
    },
]

app = FastAPI(
        openapi_tags=tags_metadata,
        docs_url='/',
        title="Advanced Machine Learning Application - Assignment 2",
        description=description,
        summary="American retailer sale revenue prediction and forecasting API",
        version="0.0.1",
        contact={
            "name": "Rohan Rocky Britto",
            "email": "rohan.r.britto@student.uts.edu.au",
        }
    )

prediction_model = load('../models/predictive/rf_pipe.joblib')
forecasting_model = load('../models/forecasting/prop_hol.joblib')

df_train = pd.read_csv('../data/sales_train.csv')
df_calendar = pd.read_csv('../data/calendar.csv')
df_events = pd.read_csv('../data/calendar_events.csv')
# Splitting weekly sell prices csv into 3 so that it can be pushed on git
split_df1 = pd.read_csv('../data/items_weekly_sell_prices_split1.csv')
split_df2 = pd.read_csv('../data/items_weekly_sell_prices_split2.csv')
split_df3 = pd.read_csv('../data/items_weekly_sell_prices_split3.csv')

# Merge the DataFrames into a single DataFrame
df_sell_prices = pd.concat([split_df1, split_df2, split_df3], ignore_index=True)

@app.get('/health/', tags=['health'], status_code=200)
def healthcheck(response:Response):
    response.status_code = 200
    return 'Welcome to the sale revenue prediction and forecasting API'

@app.get('/sales/national/', tags=['forecast'])
async def forecast(date:str):
    next_date = pd.to_datetime(date)+timedelta(days=1)
    df_test = pd.DataFrame({})
    df_test['ds'] = pd.date_range(start=next_date, periods=7)
    df_test['ds'] = df_test['ds'].dt.strftime('%Y-%m-%d')
    df_test['forecast'] = forecasting_model.predict(df_test)['yhat']

    return JSONResponse(df_test.set_index('ds').to_dict()['forecast'])

@app.get('/sales/stores/items/', tags=['predict'])
async def predict(item:str, store:str, date:str):

    df_item = df_train[(df_train['item_id']==item) & (df_train['store_id']==store)][['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].reset_index(drop=True)
    df_date = df_calendar[df_calendar['date']==date].join(df_events.set_index('date'), on='date').fillna('None').reset_index(drop=True)
    df_test = pd.concat([df_item, df_date], axis=1)
    df_test = df_test.join(df_sell_prices.set_index(['store_id', 'item_id', 'wm_yr_wk']), on=['store_id', 'item_id', 'wm_yr_wk'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_test['num_date'] = df_test['date'].dt.strftime('%Y%m%d')
    df_test['day_of_week'] = df_test['date'].dt.dayofweek
    prediction = prediction_model.predict(df_test)
    
    return JSONResponse({'prediction': prediction[0]})