import pandas as pd

DATASET = "DATASET PATH"
CLEAN_OUTPUT = "CLEAN_OUTPUT PATH"

# Loading the dataset in a dataframe
data_df = pd.read_csv(DATASET, encoding="ISO-8859-1", dtype={'Customer ID': str})

# Renaming some columns to be more representative.
data_df.rename(columns={"Price":"UnitPrice", "Customer ID":"CustomerID"}, inplace=True)

# Removing decimal parts from CustomerIDs which have no purpose; it's .0 for all customers
data_df['CustomerID'] = data_df['CustomerID'].str.split(pat='.', expand=True)[0]

# The time part in the InvoiceDate isn't neccesary for our analysis so let's remove it.
data_df['InvoiceDate'] = data_df['InvoiceDate'].str.split(pat=' ', expand=True)[0]

# Changing InvoiceDate datatype
data_df['InvoiceDate'] = pd.to_datetime(data_df['InvoiceDate'])

# Fill null values in the CustomerID column
data_df["CustomerID"].fillna("Guest", inplace=True)

# Drop invalid quantities & UnitPrices
data_df = data_df[(data_df["Quantity"] >= 1) & (data_df["UnitPrice"] > 0)]

# Add a "isGuest" column with boolean values:
data_df["isGuest"] = data_df["CustomerID"] == "Guest"

# Create a CSV file for the clean dataset
data_df.to_csv(CLEAN_OUTPUT, index=False)