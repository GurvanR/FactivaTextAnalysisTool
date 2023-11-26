import pandas as pd
from datetime import datetime
import locale
from typing import Union

import tkinter as tk
from tkinter import filedialog
import os

def convert_date_factiva(df: pd.DataFrame, column: str = "PD",  language: str = "english") -> None :
    if language == "french" :
        locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
    df[column] = df[column].apply(lambda x: datetime.strptime(x.strip(), '%d %B %Y'))
    df[column] = pd.to_datetime(df[column])


##RTFtoCSV functions 
    
# Given a path to one rtf file from factiva, return a csv file with the desired_fields.
# Here are the default fields selected : SE, HD, BY, WC, PD, ET, SN, SC, LA, CY, LP, TD, NS, RE, PUB, AN 
# Please, if you want to change them, respect the syntax : ['SE|','HD|','BY|','WC|,...]
# The name of fields are by default the name of the acronyms without the vertical bars. If
# you want to have other names, you can do it with the arg : 
# desired_fields = ['chosen_name_1', chosen_name_2', ...]


#Be careful, if the data is downloaded in english, maybe the convert date time won't work.
# have to try.


# Idea : do an app, to do so first :
# 1. rewrite it for not a path but a file directly

from striprtf.striprtf import rtf_to_text 

def FactivaRTFtoDF( rtf_file_path: str, 
                     output_format: Union['csv', pd.DataFrame] = 'DataFrame',
                     output_file_name: str = 'FactivaPostProcessed', 
                     desired_acronyms = ['SE|','HD|','BY|','WC|','PD|','ET|','SN|','SC|','LA|','CY|','LP|','TD|','NS|','RE|','PUB|','AN|'],
                     desired_fields = [], 
                     join_LP_TD: bool= True, 
                     date_time_format: bool = True, 
                     datetime_language:str = "english") -> pd.DataFrame:
    
    #Opening the rtf file
    with open(rtf_file_path, 'r') as file: 
        rtf_text = file.read() 
        plain_text = rtf_to_text(rtf_text) 

    #Formatting the text into items of each row of the rtf file.
    lines = plain_text.split('|')

    # Reconstruct the substrings by joining the split parts with '|'
    lines = [f"{lines[i].strip()}|{lines[i + 1].strip()}" for i in range(0, len(lines) - 1, 2)]
    
    # Initialize an empty list to store sequences, each sequence will correspond to an article.
    sequences = []
    current_sequence = []

    for item in lines:
        current_sequence.append(item)
        if item.startswith('AN|'):
            # Start a new sequence when encountering an item starting with 'AN|'
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = []
    
    #Creating the dictionnary.

    if not desired_fields : desired_fields = [item.replace('|', '') for item in desired_acronyms]

    data = {field: [] for field in desired_fields}

    for sequence in sequences :
        for AA, key in zip(desired_acronyms, data.keys()) : 
            field = [line for line in sequence if line.startswith(AA)]
            if field : 
                data[key].append(field[0].split('|')[1])
            else : data[key].append("Missing Data")
    
    df = pd.DataFrame(data)

    if join_LP_TD : 
        LP_pos = df.columns.get_loc('LP')
        df['Concatened Colmuns'] = df['LP'] + df['TD']
        df = df.drop(columns=['LP', 'TD'])
        LPTD = df.pop('Concatened Colmuns')
        df.insert(LP_pos, 'LPTD', LPTD)
    
    if date_time_format : convert_date_factiva(df, language=datetime_language)

    if output_format == 'csv' : 
        return df.to_csv(output_file_name + '.csv', encoding='utf-8', index=False)

    else : return df

import os

#rtf_folder_path : path of the folder containing all the rtf file 
# you want to merge in one CSV (or DataFrame) file.
# output_format: 'CSV', 'DataFrame',
#  

def FactivaRTFtoCSV( rtf_folder_path: str, 
                     output_format: Union['csv', 'DataFrame'] = 'csv',
                     output_file_name: str = 'FactivaPostProcessed', 
                     desired_acronyms = ['SE|','HD|','BY|','WC|','PD|','ET|','SN|','SC|','LA|','CY|','LP|','TD|','NS|','RE|','PUB|','AN|'],
                     desired_fields = [], 
                     join_LP_TD: bool= True, 
                     date_time_format: bool = True, 
                     datetime_language:str = "english") -> "csv":
    
    dataframes = []
    # Check if the specified path exists and is a directory
    if os.path.isdir(rtf_folder_path):
        # Iterate over files in the directory
        for file_name in os.listdir(rtf_folder_path):
            # Check if the item is a file (not a directory)
            file_path = os.path.join(rtf_folder_path, file_name)
            if os.path.isfile(file_path):
                # Check if the file has the ".rtf" extension
                if os.path.splitext(file_path)[1].lower() == '.rtf':
                    dataframes.append(FactivaRTFtoDF(rtf_folder_path + '/' + file_name, output_format, desired_acronyms, desired_fields, join_LP_TD, date_time_format, datetime_language))
    
    merged_df = dataframes[0]
    for df in dataframes[1:] :
        merged_df = pd.concat([df, merged_df], ignore_index=True)
    #merged_df = pd.concat(dataframes, ignore_index=True) Maybe works ?
    
    if output_format == 'csv' :
        merged_df.to_csv(output_file_name + '.csv', encoding='utf-8', index=False)
        print("PASSAGE ici")
        return output_file_name + '.csv'

    return merged_df


def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        output_file = os.path.join(folder_path, file_name_entry.get())
        result_file = FactivaRTFtoCSV(folder_path, output_file_name=output_file)
        if result_file:
            status_label.config(text=f"CSV file generated: {result_file}", fg="green")
        else:
            status_label.config(text="Error occurred during CSV generation", fg="red")

root = tk.Tk()
root.title("Folder to CSV Converter")

select_button = tk.Button(root, text="Select Folder", command=select_folder)
select_button.pack()

file_name_label = tk.Label(root, text="Enter output file name:")
file_name_label.pack()

file_name_entry = tk.Entry(root)
file_name_entry.pack()

status_label = tk.Label(root, text="", fg="black")
status_label.pack()

root.mainloop()
