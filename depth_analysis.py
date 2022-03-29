from turtle import width
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
import sys
import os
from plyfile import PlyData, PlyElement
import statistics
import csv
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import math
import plotly.express as px
import statsmodels.api as sm

sum = 0
mean = 0
row = 0
column = 0

data_root = sys.argv[1]    
newcsv_name = sys.argv[2]
combine_csvs = True

def main():
    global points
    counter = 0

    if combine_csvs:
        csvgt = pd.read_csv(os.path.join(data_root, 'ground-truth.csv')) # read ground truth csv

        combine_data = None # since no data initially
        for idx, row in csvgt.iterrows(): # loop through rows in ground truth csv and extract index and its corresponding row
            csv_file_path = os.path.join(data_root, f"{row['Foldername']}.csv") # read the csv of folder name in each row of ground truth csv
            csv_data = pd.read_csv(csv_file_path) # read csv with depth data
            csv_data['gt'] = row['Groundtruth']
            csv_data['Name'] = row['Foldername']
            if combine_data is None:
                combine_data = csv_data # take data from first csv
            else:
                combine_data = combine_data.append(csv_data) # add gt data from next csv files


        df = pd.DataFrame({ 'Name': combine_data['Name'],
            'x': combine_data['x'], # set x values in first column
            'y': combine_data['y'], # set y values in second column
            'd0': combine_data['d0'],
            'd1': combine_data['d1'],
            'd2': combine_data['d2'], # set depth values in third column
            'd3': combine_data['d3'],
            'd4': combine_data['d4'],
            'd5': combine_data['d5'],
            'd6': combine_data['d6'],
            'd7': combine_data['d7'],
            'gt': combine_data['gt']} 
            ) 
            
        df.to_csv(f'/mnt/sda1/zed_depth_camera/{newcsv_name}.csv', index = False, header=True)
        print(combine_data)
        print(df)

    # depth analysis
    newcsv = pd.read_csv(os.path.join(data_root, f'{newcsv_name}.csv')) # read ground truth csv

    # for i in range(8):
    #     mse = mean_squared_error(newcsv['gt'].to_list(), newcsv['d' + str(i)].to_list())

    #     rmse = math.sqrt(mse)

    #     print(rmse)

    #     r2score = r2_score(newcsv['gt'].to_list(), newcsv['d' + str(i)].to_list())
    #     print(r2score)

    mse = mean_squared_error(newcsv['gt'].to_list(), newcsv['d6'].to_list())

    rmse = math.sqrt(mse)

    print(rmse)

    r2score = r2_score(newcsv['gt'].to_list(), newcsv['d6'].to_list())
    print(r2score)
    
    #df = pd.read_csv('dataanalysis.csv')
    fig = px.scatter(newcsv, x="gt", y='d6', trendline="ols")
    rmse = np.sqrt(mean_squared_error(newcsv['gt'], newcsv['d6']))
    r2 = r2_score(newcsv['gt'], newcsv['d6'])
    print(f"rmse: {rmse}, r2: {r2}")

    fig.update_layout(
    height=800,
    title_text='Depth Analysis'
)

    fig.add_annotation(text= "RMSE" + ":" + " " + str(rmse),
                  xref="paper", yref="paper",
                  x=0.95, y=0.2, showarrow=False)

    fig.add_annotation(text= "r2 score" + ":" + " " + str(r2score),
                xref="paper", yref="paper",
                x=0.963, y=0.1, showarrow=False)

    fig.show()
    fig.write_html('result7.html')


if __name__ == "__main__":
    main()


# from 1633