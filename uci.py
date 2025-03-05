import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
retail_data = pd.read_excel(url)

retail_data['InvoiceDate'] = pd.to_datetime(retail_data['InvoiceDate'])
retail_data['Sales'] = retail_data['Quantity'] * retail_data['UnitPrice']

# Remove returns (negative quantities) and missing customer IDs
retail_data = retail_data[(retail_data['Quantity'] > 0) & (retail_data['CustomerID'].notna())]

# Aggregate to daily sales
daily_sales = retail_data.groupby(retail_data['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
daily_sales.columns = ['date', 'kpi_value']
daily_sales['date'] = pd.to_datetime(daily_sales['date'])
daily_sales.to_parquet('retail_sales_uci.parquet')
