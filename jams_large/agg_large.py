import pandas as pd
import os
import jams

from tqdm import tqdm
from json.decoder import JSONDecodeError

JAMS_DIR = '/scratch/ci411/rap_data/jams_fma_large'

metric_tags = ['Voicing Recall', 'Voicing False Alarm', 'Raw Pitch Accuracy',\
               'Raw Chroma Accuracy', 'Overall Accuracy']

cref_map = {'Voicing Recall': 'cref_recall', 'Voicing False Alarm': 'cref_false_alarm', \
            'Raw Pitch Accuracy': 'cref_pitch_acc', 'Raw Chroma Accuracy': 'cref_chroma_acc',\
            'Overall Accuracy': 'cref_overall_acc'}

pref_map = {'Voicing Recall': 'pref_recall', 'Voicing False Alarm': 'pref_false_alarm', \
            'Raw Pitch Accuracy': 'pref_pitch_acc', 'Raw Chroma Accuracy': 'pref_chroma_acc',\
            'Overall Accuracy': 'pref_overall_acc'}

def agg_jams(jams_dir):
    #define structur of output df
    jams_df = pd.DataFrame(columns=['track_id', 'cref_recall', 'cref_false_alarm', 'cref_pitch_acc',\
                                'cref_chroma_acc', 'cref_overall_acc', 'pref_recall', 'pref_false_alarm',\
                                'pref_pitch_acc', 'pref_chroma_acc', 'pref_overall_acc'])
    
    #iterate through files in directory
    for path, subdirs, files in os.walk(jams_dir):
        for file in tqdm(files):
            
            #define paths and load jams files
            
            filepath = os.path.join(jams_dir, file)

            try:
                loaded_jams = jams.load(filepath)
            except JSONDecodeError:
                print("Error loading {}".format(file))
                continue
            
            #establish dict structure and use filename to grab track id
            row_dict = {}
            row_dict['track_id'] = file.split('.')[0]
            
            #extract tags, return NaN if no voicings in reference
            for tag in metric_tags:
                if type(loaded_jams['sandbox']['crepe_ref_metrics'])==dict:
                    row_dict[cref_map[tag]] = loaded_jams['sandbox']['crepe_ref_metrics'][tag]
                else:
                    row_dict[cref_map[tag]] = float('NaN')
                if type(loaded_jams['sandbox']['pyin_ref_metrics'])==dict:    
                    row_dict[pref_map[tag]] = loaded_jams['sandbox']['pyin_ref_metrics'][tag]
                else:
                    row_dict[pref_map[tag]] = float('NaN')
            jams_df = jams_df.append(row_dict, ignore_index=True)
    
    return jams_df

fma_large_df = agg_jams(JAMS_DIR)

fma_large_df.to_csv('/scratch/ci411/rap_data/fma_large_pitchcontour.csv')

