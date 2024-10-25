from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import seaborn as sns

from models import *

def filter_races_by_years(races, start_year = 2018, end_year = 2023):
    return races[(races['year'] >= start_year) & (races['year'] <= end_year)]

def get_data():
    drivers = Drivers()
    drivers.df.drop(['url', 'dob', 'number', 'forename', 'surname'], axis=1, inplace=True)

    ps = Pit_Stops()
    ps.df.drop(['time', 'duration'], axis=1, inplace=True)

    results = Results()
    results.df.drop(['milliseconds', 'time', 'fastestLapSpeed', 'position',
        'positionText', 'rank', 'number', 'fastestLap'], axis=1, inplace=True)

    races = Races()
    races.df.drop(['time', 'url', 'date', 'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date',
        'fp3_time', 'quali_date', 'quali_time', 'sprint_date', 'sprint_time'], axis=1, inplace=True)

    constructors = Contructors()
    constructors.df.drop(['constructorRef', 'url'], axis=1, inplace=True)

    lt = Lap_Times()
    lt.df.drop(['time'], axis=1, inplace=True)

    quali = Qualifying()
    quali.df.drop(['qualifyId', 'number', 'position', 'constructorId'], axis=1, inplace=True)

    return drivers, ps, results, races, constructors, lt, quali

def clean_data(start_year = 2018, end_year = 2023, is_checkpoint = True, ps_threshold = 0.5):

    drivers, ps, results, races, constructors, lt, quali = get_data()

    races.df = filter_races_by_years(races.df, start_year, end_year)
    races.df = races.df.merge(results.df, on='raceId', how='inner')

    #----------------- Pitstop data Pre-Processing ----------------------
    ps.df = ps.df[ps.df['raceId'].isin(races.df['raceId'])]

    condition = ps.df['stop'] > 7
    ps.df.loc[condition, ['stop', 'lap']] = ps.df.loc[condition, ['lap', 'stop']].values
    
    avg_milliseconds = ps.df.groupby(['raceId', 'driverId'])['milliseconds'].mean().reset_index(name='avg_ps_milliseconds')
    max_stops = ps.df.groupby(['raceId', 'driverId'])['stop'].max().reset_index(name='max_ps_stops')
    df_pivot = ps.df.pivot(index=['raceId', 'driverId'], columns='stop', values='lap').add_prefix('lap_ps_').reset_index()
    pitStop_info_by_race = avg_milliseconds.merge(max_stops, on=['raceId', 'driverId']).merge(df_pivot, on=['raceId', 'driverId'])
    pitStop_info_by_race = pitStop_info_by_race[pitStop_info_by_race['raceId'].isin(races.df['raceId'])]

    races.df = races.df.merge(pitStop_info_by_race, on=['raceId', 'driverId'], how='inner')
    races.df.drop(['round', 'resultId'], axis=1, inplace=True)
    
    all_ps = [col for col in pitStop_info_by_race.columns if col not in ['raceId', 'driverId', 'avg_ps_milliseconds', 'max_ps_stops']]
    
    for ps in all_ps:
        nan_density = races.df[ps].isna().mean()
        if nan_density > ps_threshold:
            races.df.drop(columns=[ps], inplace=True)

    #----------------- Lap Time data Pre-Proccssing ----------------------
    lt.df = lt.df[lt.df['raceId'].isin(races.df['raceId'])]

    avg_milliseconds = lt.df.groupby(['raceId', 'driverId'])['milliseconds'].mean().reset_index(name='avg_lap_milliseconds')
    position_counts = lt.df.groupby(['raceId', 'driverId'])['position'].value_counts().unstack(fill_value=0)
    position_counts = position_counts.rename(columns={1: 'laps_in_position_1', 2: 'laps_in_position_2', 3: 'laps_in_position_3'}).reset_index()[['raceId', 'driverId', 'laps_in_position_1', 'laps_in_position_2', 'laps_in_position_3']]

    lap_info_by_race = avg_milliseconds.merge(position_counts, on=['raceId', 'driverId'], how='left')

    races.df = races.df.merge(lap_info_by_race, on=['raceId', 'driverId'], how='inner')

    def lap_time_to_milliseconds(lap_time):
        if lap_time in ['\\N', None] or pd.isna(lap_time):
            return np.nan

        minute, second_millisecond = lap_time.split(':')
        second, millisecond = second_millisecond.split('.')
        
        total_milliseconds = (int(minute) * 60 * 1000) + (int(second) * 1000) + int(millisecond)
        return total_milliseconds
    
    races.df['fastest_race_lap_ms'] = races.df['fastestLapTime'].apply(lap_time_to_milliseconds)
    races.df.drop(['fastestLapTime'], axis=1, inplace=True)

    quali.df['q1'] = quali.df['q1'].apply(lap_time_to_milliseconds)
    quali.df['q2'] = quali.df['q2'].apply(lap_time_to_milliseconds)
    quali.df['q3'] = quali.df['q3'].apply(lap_time_to_milliseconds)

    quali.df['fastest_quali_lap_ms'] = quali.df[['q1', 'q2', 'q3']].min(axis=1)
    quali.df.drop(['q1', 'q2', 'q3'], axis=1, inplace=True)
    
    races.df = races.df.merge(quali.df, on=['raceId', 'driverId'], how='inner')

    if is_checkpoint:
        races.df.to_csv('cleaned_data.csv', index=False)
    
    #----------------- Dataset by Driver by Year ----------------------
    races.df.drop(columns=['name','constructorId'], inplace=True)
    races.df = races.df.drop(columns=[col for col in races.df.columns if col.startswith('lap_ps_')])# Remove all laps ps for now.
    
    avg_max_ps_stops = races.df.groupby(['year', 'driverId'])['max_ps_stops'].mean().reset_index(name='avg_max_ps_stops')
    laps_in_positions_sum = races.df.groupby(['year', 'driverId'])[['laps_in_position_1', 'laps_in_position_2', 'laps_in_position_3']].sum().reset_index()

    total_laps_per_year = races.df.groupby(['raceId', 'year'])['laps'].max().reset_index().groupby('year')['laps'].sum().reset_index()
    laps_in_position_1_ratio = laps_in_positions_sum[['year', 'driverId', 'laps_in_position_1']].merge(total_laps_per_year, on='year', how='left')
    laps_in_position_1_ratio['lead_lap_ratio'] = laps_in_position_1_ratio['laps_in_position_1'] / laps_in_position_1_ratio['laps']
    laps_in_position_1_ratio.drop(columns=['laps_in_position_1','laps'], inplace=True)

    num_of_accidents = races.df[races.df['statusId'].isin([3, 4, 104])].groupby(['year', 'driverId']).size().reset_index(name='num_of_accidents')

    team_related_status = [5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 46, 47, 48,
        49, 51, 56, 129, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 
        95, 96, 98, 99, 121, 126, 131, 132, 135, 136, 137, 138, 140, 141]
    status_counts_2 = races.df[races.df['statusId'].isin(team_related_status)].groupby(['year', 'driverId']).size().reset_index(name='team_related_dnf')

    driver_related_status = [20, 41, 31, 54, 59, 60, 64, 65, 81, 82, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 130, 139]
    status_counts_3 = races.df[races.df['statusId'].isin(driver_related_status)].groupby(['year', 'driverId']).size().reset_index(name='driver_related_dnf')

    avg_grid = races.df.groupby(['year', 'driverId'])['grid'].mean().reset_index(name='avg_grid')

    sum_points = races.df.groupby(['year', 'driverId'])['points'].sum().reset_index(name='total_points')

    df = avg_max_ps_stops.merge(laps_in_positions_sum, on=['year', 'driverId'], how='left')
    df = df.merge(laps_in_position_1_ratio, on=['year', 'driverId'], how='left')
    df = df.merge(num_of_accidents, on=['year', 'driverId'], how='left').fillna(0)
    df = df.merge(status_counts_2, on=['year', 'driverId'], how='left').fillna(0)
    df = df.merge(status_counts_3, on=['year', 'driverId'], how='left').fillna(0)
    df = df.merge(avg_grid, on=['year', 'driverId'], how='left')
    df = df.merge(sum_points, on=['year', 'driverId'], how='left')
    
    return df

if __name__ == '__main__':
    df = clean_data(2017, 2022)
    df.head()