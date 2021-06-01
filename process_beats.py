import numpy as np
import librosa as lr
import os
import sys
from tqdm import tqdm
import argparse
from random import shuffle

import jams
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

import jams
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    
    parser.add_argument('--jobs', dest='n_jobs', type=int,
                        default=2,
                        help='Number of jobs to run in parallel')

    parser.add_argument('--audio-path', dest='audio_path', type=str,
                       default='/scratch/work/sonyc/marl/private_datasets/FMA/fma_small/fma_small/000',
                       help='Directory containing target audio files')
    
    parser.add_argument('--jams-dir', dest='jams_dir', type=str,
                       default='/scratch/ci411/rap_data/jams_test_000',
                       help='Directory containing jams files')
    
    parser.add_argument('--shuffle', dest='shuffle', type=bool,
                        default=True,
                        help='Option to process files in a random order')

    
    return parser.parse_args(args)

def process_downbeats(audiopath, jamspath, prob_proc, beat_proc):
    jams_data = jams.load(jamspath)
    if len(jams_data.annotations['beat'])>0:
        print("Jams {} has beats already!".format(jamspath))
        return
    beat_probs = prob_proc(audiopath)
    beat_preds = beat_proc(beat_probs)
    
    
    beat_ann = jams.Annotation(namespace='beat')
    for beat in beat_preds:
        beat_ann.append(time=beat[0], duration=0., value=beat[1])
    
    jams_data.annotations.append(beat_ann)
    jams_data.save(jamspath)
    

if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    
    prob_proc = RNNDownBeatProcessor()
    beat_proc = DBNDownBeatTrackingProcessor(beats_per_bar=4, min_bpm=60, max_bpm=180, fps=100)

    audio_path = params.audio_path
    jams_dir = params.jams_dir

    path_pairs = []
    
    
    for path, subdirs, files in os.walk(audio_path):
        if params.shuffle:
            shuffle(files)
        for file in tqdm(files):
            inpath = os.path.join(audio_path, file)
            songname = '.'.join(file.split('.')[:-1])
            jams_path = os.path.join(jams_dir, songname+'.jamz')
            path_pairs.append((inpath, jams_path))    
    
    for inpath, outpath in tqdm(path_pairs):
        process_downbeats(inpath, outpath, prob_proc, beat_proc)
        