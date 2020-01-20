import networkx as nx
import pandas as pd

calls = pd.read_csv('data/calls.csv', sep=';', dtype=int)
sms = pd.read_csv('data/sms.csv', sep=';', dtype=int)
print(calls.head())
print(sms.head())

np_calls = calls.values
print(f'np_calls.shape: {np_calls.shape}')
print(f'np_sms.shape: {sms.values.shape}')
