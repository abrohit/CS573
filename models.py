import pandas as pd
import numpy as np

class Models():
    def __init__(self):
        pass
    
    def join(self, df:pd.DataFrame, on:str, how:str = 'inner', inplace:bool = True):
        if inplace:
            self.df = self.df.merge(df, on=on, how=how)
            return None
        else:
            return self.df.merge(df, on=on, how=how)

class Circuits(Models):
    def __init__(self, filepath:str = './database/circuits.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Contructor_Results(Models):
    def __init__(self, filepath:str = './database/constructor_results.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Contructor_Standing(Models):
    def __init__(self, filepath:str = './database/constructor_standings.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Contructors(Models):
    def __init__(self, filepath:str = './database/constructors.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Driver_Standings(Models):
    def __init__(self, filepath:str = './database/driver_standings.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Drivers(Models):
    def __init__(self, filepath:str = './database/drivers.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Lap_Times(Models):
    def __init__(self, filepath:str = './database/lap_times.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Pit_Stops(Models):
    def __init__(self, filepath:str = './database/pit_stops.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Qualifying(Models):
    def __init__(self, filepath:str = './database/qualifying.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Races(Models):
    def __init__(self, filepath:str = './database/races.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Results(Models):
    def __init__(self, filepath:str = './database/results.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Seasons(Models):
    def __init__(self, filepath:str = './database/seasons.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Sprint_Results(Models):
    def __init__(self, filepath:str = './database/sprint_results.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

class Status(Models):
    def __init__(self, filepath:str = './database/status.csv'):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

if __name__ == '__main__':
    circuits = Circuits()
    cr = Contructor_Results()
    cs = Contructor_Standing()
    
    contructors = Contructors()

    cr.join(contructors.df, on='constructorId')
    print(cr.df.head())

    d_standings = Driver_Standings()
    drivers = Drivers()

    lt = Lap_Times()
    pt = Pit_Stops()

    qualifying = Qualifying()

    races = Races()
    results = Results()
    
    seasons = Seasons()

    sr = Sprint_Results()

    status = Status()
