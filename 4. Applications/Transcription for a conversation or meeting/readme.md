# Transcribe a meeting

we’ll use the Whisper model to generate a transcription for a conversation or meeting between two or more speakers. We’ll then pair it with a speaker diarization model to predict “who spoke when”. By matching the timestamps from the Whisper transcriptions with the timestamps from the speaker diarization model, we can predict an end-to-end meeting transcription with fully formatted start / end times for each speaker.

<img src='https://huggingface.co/datasets/huggingface-course/audio-course-images/resolve/main/diarization_transcription.png'>


## 1. Speaker Diarization
Speaker diarization (or diarisation) is the task of taking an unlabelled audio input and predicting “who spoke when”. In doing so, we can predict start / end timestamps for each speaker turn, corresponding to when each speaker starts speaking and when they finish.

## 2. Speech transcription
Get Speech transcription using Whisper pre-trained model

# 3. Speechbox
To get the final transcription, we’ll align the timestamps from the diarization model with those from the Whisper model. The diarization model predicted the first speaker to end at 14.5 seconds, and the second speaker to start at 15.4s, whereas Whisper predicted segment boundaries at 13.88, 15.48 and 19.44 seconds respectively. Since the timestamps from Whisper don’t match perfectly with those from the diarization model, we need to find which of these boundaries are closest to 14.5 and 15.4 seconds, and segment the transcription by speakers accordingly. Specifically, we’ll find the closest alignment between diarization and transcription timestamps by minimising the absolute distance between both.
