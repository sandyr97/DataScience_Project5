import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md

df=pd.read_csv("../data/historicalProductDemand.csv",infer_datetime_format = True)
df.head(10)


# # Clean Dataset:
df.dropna(inplace=True)


# ## Chosing a Product:
df['freq'] = df.groupby('Product_Code')['Product_Code'].transform('count')
df.head(10)
print(df.sort_values(by='freq',ascending=False).head(1))

df_Product=df[df.Product_Code == "Product_1359"]
df_Product = df_Product.reset_index(drop=True)
print(df_Product.head(10))

df_Product['Order_Demand'] = df_Product['Order_Demand'].str.replace(r"\(.*\)","")
print(df_Product.head(10))
df_Product['Date'] = pd.to_datetime(df_Product['Date'])
print (df_Product[pd.to_datetime(df_Product['Date'], errors='coerce').isnull()])
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_Product['Order_Demand'] = le.fit_transform(df_Product["Order_Demand"].astype(str))
df_Product['Code'] = df_Product['Product_Code'].map(lambda x: x.split('_', 1)[-1])
df_Product['Warehouse'] = df_Product['Warehouse'].map(lambda x: x.split('_', 1)[-1])
df_Product['Category'] = df_Product['Product_Category'].map(lambda x: x.split('_', 1)[-1])
df_Product['Code'] = df_Product['Code'].astype(int)
df_Product['WarehouseLE'] = le.fit_transform(df_Product["Warehouse"].astype(str))
df_Product['CategoryLE'] = le.fit_transform(df_Product["Category"].astype(str))

print(df_Product.dtypes)
df_Product['day'] = pd.DatetimeIndex(df_Product['Date']).day
df_Product['month'] = pd.DatetimeIndex(df_Product['Date']).month
df_Product['year'] = pd.DatetimeIndex(df_Product['Date']).year
dt=df_Product['Date']
dt = pd.DatetimeIndex ( dt ).astype ( np.int64 )/1000000
df_Product['unixTime']=dt
df_Product.head(10)

# # Visualization
# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='Order_Demand', data=df_Product, ax=axes[0])
sns.boxplot(x='month', y='Order_Demand', data=df_Product.loc[~df_Product.year.isin([2014, 2016]), :])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18);
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()

fig = plt.figure()
ax= fig.add_subplot(1,1,1)
ax.scatter(df_Product['unixTime'],df_Product['Order_Demand'], marker='.')
plt.xlim(1.40e+12, 1.4075e+12)
ax.set_ylabel('Order Demand');
ax.set_xlabel('Date');

avgProd=df_Product.groupby('Date', as_index=False).Order_Demand.mean()
dt=avgProd['Date']
dt = pd.DatetimeIndex ( dt ).astype ( np.int64 )/1000000
avgProd['unixTime']=dt
print(avgProd.head(10))
avgProd = avgProd.set_index('Date')

fig = plt.figure()
ax= fig.add_subplot(1,1,1)
ax.scatter(avgProd['unixTime'],avgProd['Order_Demand'], marker='.')
plt.xlim(1.40e+12, 1.4075e+12)
ax.set_ylabel('Order Demand');
ax.set_xlabel('Date');
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xticklabels(avgProd.index.values, rotation=25)
ax.plot_date(x=avgProd.index.values, y=avgProd.Order_Demand, ls='-', marker='|')
ax=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(xfmt)
plt.title("Order Demand Line Plot")
plt.xlabel("Date")
plt.ylabel("Order Demand")
df_Product['Date'].value_counts().sort_values().plot.line()
plt.title("Date Line Plot")
plt.xlabel("Date")
plt.ylabel("Frequency")

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(avgProd['Order_Demand'])
plt.ylim(-.1,.1)
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
series = pd.Series(avgProd['Order_Demand'])
result = seasonal_decompose(series, model='multiplicative', freq=100)
result.plot()
plt.show()

from scipy.signal import savgol_filter

fig = plt.figure()
ax= fig.add_subplot(1,1,1)
x=avgProd.index.values
y=avgProd['Order_Demand']
yhat = savgol_filter(y, 51, 3)

ax=plt.gca()
xfmt = md.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(xfmt)
plt.xticks( rotation=25 )
ax.set_ylabel('Order Demand');
ax.set_xlabel('Date');
plt.title("Smoothing Order Demand Plot")

plt.plot(x,y)
plt.plot(x,yhat, color='red')
plt.show()

from sklearn.model_selection import train_test_split
labels = df_Product['Order_Demand']
features = df_Product[['year' , 'month', 'Code', 'day', 'WarehouseLE', 'CategoryLE']]
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    test_size=0.20,
                                                    random_state=42)

from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor()
model = GBR.fit(X_train, y_train.values.ravel())
prediction = model.predict(X_test)
print(GBR.score(X_test, y_test))
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor()
model = RFR.fit(X_train, y_train.values.ravel())
prediction = model.predict(X_test)
print(RFR.score(X_test, y_test))
