import pandas_datareader.data as web

start_date = '2012-01-01'
end_date = '2022-12-31'
retail_sales = web.DataReader('RSXFS', 'fred', start_date, end_date)
retail_sales = retail_sales.reset_index()
retail_sales.columns = ['date', 'kpi_value']
retail_sales.to_parquet('retail_sales_fred.parquet')

unemployment_claims = web.DataReader('ICSA', 'fred', start_date, end_date)
unemployment_claims = unemployment_claims.reset_index()
unemployment_claims.columns = ['date', 'kpi_value']
unemployment_claims.to_parquet('unemployment_claims.parquet')
