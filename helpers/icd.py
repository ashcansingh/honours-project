import pandas as pd
def create_csv():
    # Absolute Path
    path_to_icd = "/research/home/he231839/honours/GitHub2/honours-project/ICD-10 Codes/icd10cm-codes-2025.txt"

    with open(path_to_icd, 'r') as icd:
        all_icd = icd.read().strip().split('\n')

    icd_split = [icd.split(' ', 1) for icd in all_icd] # Split into [ICD, Description] 
    icd_clean = [[icd.strip() for icd in icd_pair] for icd_pair in icd_split] # Strip trailing spaces in strings

    pd.DataFrame