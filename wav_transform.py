import os
from pydub import AudioSegment


def convert_opus_to_wav(input_folder, output_folder, sample_rate=16000):
    """
    Convert all .opus files in a folder to .wav format.

    Args:
        input_folder (str): Path to folder containing .opus files.
        output_folder (str): Path to save .wav files.
        sample_rate (int): Target sample rate (default: 16000 Hz).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".opus"):
            opus_path = os.path.join(input_folder, filename)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_path = os.path.join(output_folder, wav_filename)

            # Convert using pydub (requires ffmpeg)
            try:
                audio = AudioSegment.from_file(opus_path, codec="opus")
                audio = audio.set_frame_rate(sample_rate)
                audio.export(wav_path, format="wav")
                print(f"Converted: {filename} → {wav_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {str(e)}")


# Example usage
input_folder = r"C:\Users\hp\Desktop\achraf___" # Replace with your folder
output_folder = r"C:\Users\hp\Desktop\achraf_wav"  # Replace with output folder
convert_opus_to_wav(input_folder, output_folder)