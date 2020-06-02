from pydub import AudioSegment

def overlawAudio(file1, file2):
    sound1 = AudioSegment.from_file(file1)
    sound2 = AudioSegment.from_file(file2)

    combined = sound1.overlay(sound2)

    combined.export("input.wav", format='wav')
    return 'finished'
