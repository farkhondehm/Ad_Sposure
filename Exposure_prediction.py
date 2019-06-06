==============================================================================================================================================
### 1. Import Library
==========================================================================================================================================
import gzip
import numpy as np
import pandas as pd
import matplotlib as plt
import gmaps
import gmaps.datasets
import gmaps.geojson_geometries
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os
os.environ["PROJ_LIB"] = "C:/Users/User/Anaconda3/pkgs/proj4-5.2.0-ha925a31_1/Library/share"
import gmplot
#import basemap
from mpl_toolkits import basemap
from mpl_toolkits.basemap import Basemap
import requests 
import datetime
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio
import plotly
import os
import numpy as np
init_notebook_mode(connected=True)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression

=========================================================================================================================================
### 2. Upload Data and Cleaning Data
=========================================================================================================================================
# Read Data
data=pd.read_csv('C:/Users/User/Downloads/dump.csv/select___from_scans__where_device_in_965.csv', sep=',',  encoding='latin1', engine='python')
Population=pd.read_excel(r'C:\\Users\\User\\Desktop\\in.xlsx')  ### poulation of different cities in ontario

# Prepare Data
data['ts']=data['deafult_ts'].astype('datetime64[ns]')
data['weekday']=data['ts'].dt.weekday_name  ### change the weekday from integer to string name
data=data[pd.notnull(f_new['city'])] ### remove null data
data= data.set_index(['ts'])
data.drop(['id','oui', 'ssid','rssi', 'channel', 'created', 'kinesis_ts', 'deafult_ts'], axis=1, inplace=True) # drop columns with no information

# mapping device to truck
dic= {965: 29, 2808:29, 2859: 29, 2791:29, 
      945: 35, 2739: 35,
      2848:59, 1016:59, 1237:59, 1007:59, 2691: 59, 1098:59, 1209: 59,
      1348: 82, 1128:82, 2849: 82}

data['device'].replace(dic, inplace=True)
data = data.astype({"device": int})

============================================================================================================================================
## 3. Explarotary Data Analysis
==========================================================================================================================================
plt.rcParams['figure.figsize']=(8,4)

Exposure_hr=df.groupby(['hour'])['mac_addr'].count()
Exposure_dataframe=pd.DataFrame({'hr': Exposure_hr.index, 'imp': Exposure_hr.values})
x =Exposure_dataframe.hr
y = Exposure_dataframe.imp
data = [go.Bar(x=x, y=y)]

layout = go.Layout(
    title='',
    
    xaxis=dict(title='Hour',tickfont=dict(size=13), titlefont=dict(size=27), tickmode='linear'), 
    
    yaxis=dict(title='No of People Exposed to Ads.',tickfont=dict(size=20), titlefont=dict(size=20)))
    
fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='style-bar')

if not os.path.exists('images'):
    os.mkdir('images')
    
plotly.offline.plot(fig, filename='hourly_exposure.html') 

####################################
# 3.1 Different cities exposure
####################################
data.groupby(['city'])['mac_addr'].count().plot.bar() 

========================================================================================================================================
### 4.1 Generate Heat Map for one Truck on a certain day (April 29, 2019)
========================================================================================================================================
April_29_data=data[(data.month==4)& (data.day==29)&(data.device=='82')]

location =April_29_data.iloc[::10, 0:2]
from folium.plugins import HeatMap

from folium import plugins
import folium
location=location.values.tolist()
for coord in location:
    mapit = folium.Map(location=[coord[1], coord[0]], zoom_start=10)

location = location =April_29_data.iloc[::10, 0:2]
heat_data = [[row['lat'],row['lon']] for index, row in location.iterrows()]
HeatMap(heat_data, max_intensity=30, point_radius=4, dissipating = True ).add_to(mapit)
mapit.save( 'April_29_data_route.html')

========================================================================================================================================
### 5. Calculate Growth Rating Point 
========================================================================================================================================
# GRP cal: % of poulation have been reached to Ads

def GRP(City, Month, Day, Truck):
    
    data_subset=data[(data.city==  City) & (data.month== Month) & (data.day == Day)]
    
    data_subset=data_subset.groupby(['device'])['mac_addr'].count().to_frame().reset_index() # number of impresson per truck (per particular campain)
    data_subset.set_index('device')
    dat_subset[data_subset['device']==Truck]['mac_addr']
    Population.set_index('city')
    p=Population[Population['city']==City].set_index('city')['pop'][City]  ### population of city chosen
    return dta_subset[data_subset['device']==Truck]['mac_addr'][0]/p* 100  ### unmebr of exposure divided by poulation

GRP('Toronto', 4, 22, 29)

=========================================================================================================================================
### 6. Model Prediction
=========================================================================================================================================

# 6.1 Reading data prepared for modeling by taking hourly aggregation of whole dataset
speed_mac=pd.read_excel(r'C:\\Users\\User\\Desktop\\speed_macadr_dev.xlsx')
speed_mac['real_speed']=speed_mac['speed']*3600/1000

####################################################
# 6.2 Categorize truck speed to 6 classes
####################################################
speed_mac["speed_cat"] = np.ceil(speed_mac["real_speed"] / 20)
# Label those above 6 as 6
speed_mac["speed_cat"].where(speed_mac["speed_cat"] < 6, 6.0, inplace=True)
speed_mac["speed_cat"].value_counts()

###############################################################
# 6.3 Categorize target variable (Impression or exposure/hr)
###############################################################
speed_mac["mac_addr"] = np.ceil(speed_mac["mac_addr"] /1000)
# Label those above 6 as 6
speed_mac["mac_addr"].where(speed_mac["mac_addr"] < 6, 6.0, inplace=True)
speed_mac["mac_addr"].value_counts()

############################################
# 6.4 Prepare traing and testing set
############################################
truck = speed_mac.drop(["mac_addr", 'speed', 'real_speed', 'day'], axis=1) # drop labels for training set
truck_labels = speed_mac["mac_addr"].copy()

from future_encoders import ColumnTransformer
from future_encoders import OneHotEncoder

cat_attribs = ['city','month','weekday','hour','device','speed_cat']
full_pipeline = ColumnTransformer([("cat", OneHotEncoder(), cat_attribs)])

truck_hot1 = full_pipeline.fit_transform(truck)
truck_prepared=pd.DataFrame(truck_hot1.todense())

truck_data=pd.concat([truck_prepared, truck_labels], axis=1)
X_train, X_test, y_train, y_test = train_test_split(truck_data, truck_labels, test_size=0.3, random_state=42)

##################################################################
# 6.5  Apply Random over classification for imbalanced classes
##################################################################
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_tr_ros, y_tr_ros = ros.fit_sample(X_train, y_train)
Y=pd.DataFrame(y_tr_ros, columns=['mac_addr'])
Y.mac_addr.value_counts()
X_tr_ros=pd.DataFrame(X_tr_ros) 
y_tr_ros=pd.Series(y_tr_ros)

####################################################################
# 6.6 Build the model and train it in KFold
####################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_tr_ros,y_tr_ros)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    
    X_tr, X_val = X_tr_ros.loc[trn_idx], X_tr_ros.loc[val_idx]
    y_tr, y_val = y_tr_ros.iloc[trn_idx], y_tr_ros.iloc[val_idx]
    
    rf_model = RandomForestClassifier(random_state=42).fit(X_tr, y_tr)
    y_pred = rf_model.predict(X_val)
    print('Accuracy Score: %', 100*accuracy_score(y_val, y_pred))

###############################################
# 6.7 test model on test set for validation
###############################################
y_pred = rf_model.predict(X_test)
print('Accuracy Score: %', 100*accuracy_score(y_test, y_pred))

