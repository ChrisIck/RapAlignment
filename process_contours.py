import numpy as np
import librosa as lr
import os
import sys
from tqdm import tqdm
import argparse
from random import shuffle

import spleeter
from spleeter import SpleeterError
from spleeter.separator import Separator
from spleeter.audio.adapter import get_default_audio_adapter

import crepe

import mir_eval
from mir_eval import melody

import jams

def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    
    parser.add_argument('--jobs', dest='n_jobs', type=int,
                        default=2,
                        help='Number of jobs to run in parallel')

    parser.add_argument('--audio-path', dest='audio_path', type=str,
                       default='/scratch/work/sonyc/marl/private_datasets/FMA/fma_small/fma_small/000',
                       help='Directory containing target audio files')
    
    parser.add_argument('--out-path', dest='out_path', type=str,
                       default='/scratch/ci411/rap_data/jams_test_000',
                       help='Directory containing target audio files')
    
    parser.add_argument('--overwrite', dest='overwrite', type=bool,
                       default=False,
                       help='Option to overwrite existing jams (skips otherwise)')
    
    parser.add_argument('--shuffle', dest='shuffle', type=bool,
                        default=True,
                        help='Option to process files in a random order')

    
    return parser.parse_args(args)

def drop_unvoiced(time, frequency, voicings):
    voiced_idx = np.where(voicings)
    return time[voiced_idx], frequency[voiced_idx]
    
separator = Separator('spleeter:2stems')

def process_contours(inpath, outpath, audio_loader=get_default_audio_adapter()):
    #load audio and separate vox
    try:
        x_t, sr = audio_loader.load(inpath)
    except(SpleeterError):
        print("Bad file: {}".format(inpath))
        with open('/home/ci411/RapAlignment/jams_large/bad_files.txt', 'a+') as f:
            f.write(inpath+'\n')
        return
    
    sr=int(sr)

    #use separator for vox/accompaniment and extract vox
    prediction_test = separator.separate(x_t)
    vox_t = lr.to_mono(prediction_test['vocals'].T)
    
    #extract pyin_curves
    frame_length = 2048
    hop_length = frame_length//4
    pyin_f0, pyin_vox_flag, pyin_vox_prob = lr.pyin(vox_t, lr.note_to_hz('C2'), lr.note_to_hz('C7'),\
                                                    sr=sr, frame_length=frame_length, hop_length=hop_length)
    pyin_freq, pyin_voc = melody.freq_to_voicing(pyin_f0, voicing=pyin_vox_flag)
    pyin_time = melody.constant_hop_timebase(hop_length, hop_length*(len(pyin_f0)-1))/sr
    
    #extract crepe predictions
    crepe_time, frequency, confidence, activation = crepe.predict(vox_t, sr, verbose=0)
    threshold = 0.5
    crepe_vox_flag = confidence>threshold
    crepe_freq, crepe_voc = melody.freq_to_voicing(frequency, voicing=crepe_vox_flag)
    
    #compute comparison metrics

    pyin_ref_time, pyin_ref_freq = drop_unvoiced(pyin_time, pyin_freq, pyin_voc)
    if len(pyin_ref_time>0):
        pyin_ref_results = melody.evaluate(pyin_ref_time, melody.hz2cents(pyin_ref_freq), crepe_time,\
                                           melody.hz2cents(crepe_freq), est_voicing=crepe_voc)
        pyin_ref_results = dict(pyin_ref_results)
    else:
        pyin_ref_results = float('NaN')

    #crepe as reference
    crepe_ref_time, crepe_ref_freq = drop_unvoiced(crepe_time, crepe_freq, crepe_voc)
    if len(crepe_ref_time)>0:    
        crepe_ref_results = melody.evaluate(crepe_ref_time, melody.hz2cents(crepe_ref_freq), pyin_time,\
                                            melody.hz2cents(pyin_freq), est_voicing=pyin_voc)
        crepe_ref_results = dict(crepe_ref_results)
    else:
        crepe_ref_results = float('NaN')

    all_ref_results = {'pyin_ref_metrics':pyin_ref_results, 'crepe_ref_metrics':crepe_ref_results}

    
    #save results to jams
    jam = jams.JAMS()
    jam.sandbox = all_ref_results

    track_duration = lr.get_duration(y=x_t, sr=sr)
    jam.file_metadata.duration = track_duration

    pitch_ann = jams.Annotation(namespace='pitch_contour')

    for i in range(len(pyin_freq)):
        value = {'index':0, 'frequency': pyin_freq[i], 'voiced':bool(pyin_voc[i])} 
        pitch_ann.append(time=pyin_time[i], value=value, duration=0, confidence=pyin_vox_prob[i])

    for i in range(len(crepe_freq)):
        value = {'index':1, 'frequency': crepe_freq[i], 'voiced':bool(crepe_voc[i])} 
        pitch_ann.append(time=crepe_time[i], value=value, duration=0, confidence=confidence[i])

    jam.annotations.append(pitch_ann)
    jam.save(outpath)
    

if __name__ == '__main__':
    params = process_arguments(sys.argv[1:])
    
    #build separator    
    audio_path = params.audio_path
    output_path = params.out_path

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    path_pairs = []
    
    audio_loader = get_default_audio_adapter()
    
    for path, subdirs, files in os.walk(audio_path):
        if params.shuffle:
            shuffle(files)
        for file in tqdm(files):
            inpath = os.path.join(audio_path, file)
            songname = file.split('.')[0]
            outpath = os.path.join(output_path, songname+'.jamz')
            path_pairs.append((inpath, outpath))    
    
    for inpath, outpath in tqdm(path_pairs):
        if params.overwrite:
            process_contours(inpath, outpath, audio_loader=audio_loader)
        else:
            if not os.path.exists(outpath):
                process_contours(inpath, outpath, audio_loader=audio_loader)
            else:
                print("{} exists, skipping".format(outpath))
