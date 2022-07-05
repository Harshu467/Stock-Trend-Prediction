import numpy as np
import panda as pd
import matplotlib.pyplot as  plt
import pandas_datareader as data
start = '2010-01-01'
end = '2019-12-31'

df = data.DataReader('TSLA' , 'yahoo', start , end)
df.head()