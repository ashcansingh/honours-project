import pandas as pd

def read_mimic():
    path_to_mimic_folder = '/research/home/he231839/alligator/data/mimic-iv-note'
    discharge_file = '/discharge.csv'

    mimic = pd.read_csv(path_to_mimic_folder + discharge_file)

    return mimic
        
def get_note(data, index, subject_id = None):
    if subject_id == None:
        return data.iloc[index, :]["text"]

    return discharge[discharge['subject_id'] == subject_id].iloc[0, :]["text"]