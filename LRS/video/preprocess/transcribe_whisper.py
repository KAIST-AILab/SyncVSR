# encoding: utf-8
import re
import warnings
import os
import sys
from multiprocessing import Pool, set_start_method

from tqdm import tqdm
import torch

import whisperx
from pydub import AudioSegment

from utils import pydub_to_np

default_asr_options =  {
    "beam_size": 5,
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1,
    "no_repeat_ngram_size": 0,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "prompt_reset_on_temperature": 0.5,
    "initial_prompt": None,
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 0.0,
    "word_timestamps": False,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    "suppress_numerals": False,
    "max_new_tokens": None,
    "clip_timestamps": None,
    "hallucination_silence_threshold": None,
}
default_asr_options["suppress_numerals"] = True

# https://github.com/m-bain/whisperX/blob/78dcfaab51005aa703ee21375f81ed31bc248560/whisperx/asr.py#L259
model = whisperx.load_model(
    "medium",
    "cuda",
    device_index=0,
    compute_type="float16",
    asr_options=default_asr_options,
)
model_a, metadata = whisperx.load_align_model(language_code="en", device="cuda")

def preprocess(mp4_name, model=model, model_a=model_a, metadata=metadata):
    try:
        audio = AudioSegment.from_file(mp4_name, "mp4")
        np_audio, sample_rate = pydub_to_np(audio)
        assert sample_rate == 16000

        np_audio = np_audio.squeeze()
        audio_len = len(np_audio)

        # EXCEPTION HANDLING IF AUDIO IS CRIPPLED
        if audio_len < 16000:
            return

        result = model.transcribe(np_audio, batch_size=1)

        # EXCEPTION FOR THE OTHER LANGUAGES & BLANK CASE
        if result["language"] != "en":
            return
        if len(result['segments']) == 0:
            return 
        
        # GET CAPTION
        caption_text = result['segments'][0]['text']
        caption_text = caption_text.strip().upper()
        caption_text = re.sub("[^A-Za-z0-9 ']+", "", caption_text)

        PIECE = f"""Text:  {caption_text}\nConf:  ?\n\n"""

        # EXCEPTION FOR THE LENGTH: IF 6.2 second or more, align. ELSE pass.
        if audio_len / sample_rate > 6.0:
            PIECE += """WORD START END ASDSCORE\n"""
            result = whisperx.align(result["segments"], model_a, metadata, np_audio, "cuda", return_char_alignments=False)

            for segment in result["segments"]:
                for word_dict in segment["words"]:
                    if 'start' not in word_dict:
                        continue
                    word = word_dict['word'].strip().upper()
                    word = re.sub("[^A-Za-z0-9 ']+", "", word)
                    PIECE += f"{word} {round(word_dict['start'], 2)} {round(word_dict['end'], 2)} {round(word_dict['score'], 1)}"
                    PIECE += "\n"
        else:
            pass
        
        # SAVE
        savename = mp4_name.replace(".mp4", ".txt")
        with open(savename, 'w') as txtfile_write:
            txtfile_write.write(PIECE)
        
        del np_audio, result
        torch.cuda.empty_cache()
        return
    except:
        return

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    filename = sys.argv[1]
    
    with open(filename) as txtfile:
        lines = [line.strip() for line in txtfile]
    
    all_files = lines

    set_start_method('spawn', force=True)

    num_cpus = os.cpu_count()
    print("Number of cpus: {}".format(num_cpus))
    batch_size = 8

    print(f"original pkl: {len(all_files)}")

    with Pool(batch_size) as p:
        print("mapping ...")
        results = tqdm(p.imap(preprocess, all_files), total=len(all_files))
        print("running ...")
        list(results)  # fetch the lazy results
        print("done")