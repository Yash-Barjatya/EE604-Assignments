import cv2
import numpy as np
import librosa

def count_yellow_pixels(spec_db):
    """Counts the number of "yellow" pixels in the higher freq and lower freq range.

    Args:
        spec_db: The Mel spectrogram data in the form of a 2D array (spec_db)

    Returns:
        the ratio of higher half count to lower half
    """
    # Define the dB range for "yellow"/high frequency range pixels
    lower_db = -15
    upper_db = 0

    # Create a mask for "yellow" pixels
    mask = (spec_db >= lower_db) & (spec_db <= upper_db)

    # Count "yellow" pixels in the lower and higher frequency ranges, where to divide the image into upper and lower half 45 has been choosen as the middle value 
    lower_half_mask = mask[:45, :]
    higher_half_mask = mask[45:, :]

    lower_half_count = np.count_nonzero(lower_half_mask)
    higher_half_count = np.count_nonzero(higher_half_mask)
    
    # edge case: if no lower_half_count then simply return high ratio (here 1 which would mean metal )as had it been cardboard then lower_half_count wouldn't have been zero
    if lower_half_count ==0:
        return 1
    return higher_half_count/lower_half_count

def solution(audio_path):
    ############################
    ############################
    """Classifies the quality of a brick sound based on its spectrogram.

    Args:
        audio_path: The path to the audio file.

    Returns:
        A string indicating the quality of the brick sound: "metal" or "cardboard".
    """
    ############################
    ############################

     # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Define parameters
    n_fft = 2048  # FFT points, adjust as needed
    hop_length = 512  # Sliding amount for windowed FFT, adjust as needed
    fmax = 22000  # Maximum frequency to consider
    
    # Calculate the Mel spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=n_fft, hop_length=hop_length, fmax=fmax)

    # Convert to decibels (dB)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    #count_yellow_pixels
    r_ratio=count_yellow_pixels(spec_db)
    # print('ratio of yellow high freq to low freq',r_ratio)

    if r_ratio>0.0067:
        class_name='metal'
    else:
        class_name='cardboard'
        
    ############################
    ############################
    return class_name
