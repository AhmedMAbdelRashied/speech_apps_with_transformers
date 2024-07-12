# Speech-to-speech translation
Speech-to-speech translation (STST or S2ST) is a relatively new spoken language processing task. It involves translating speech from one langauge into speech in a different language:
<img src='https://huggingface.co/datasets/huggingface-course/audio-course-images/resolve/main/s2st.png'>

STST can be viewed as an extension of the traditional machine translation (MT) task: instead of translating text from one language into another, we translate speech from one language into another. STST holds applications in the field of multilingual communication, enabling speakers in different languages to communicate with one another through the medium of speech.

we’ll explore a cascaded approach to STST, piecing together the knowledge you’ve acquired in Units 5 and 6 of the course. We’ll use a speech translation (ST) system to transcribe the source speech into text in the target language, then text-to-speech (TTS) to generate speech in the target language from the translated text:
<img src='https://huggingface.co/datasets/huggingface-course/audio-course-images/resolve/main/s2st_cascaded.png'>

We could also have used a three stage approach, where first we use an automatic speech recognition (ASR) system to transcribe the source speech into text in the same language, then machine translation to translate the transcribed text into the target language, and finally text-to-speech to generate speech in the target language. However, adding more components to the pipeline lends itself to error propagation, where the errors introduced in one system are compounded as they flow through the remaining systems, and also increases latency, since inference has to be conducted for more models.

While this cascaded approach to STST is pretty straightforward, it results in very effective STST systems. The three-stage cascaded system of ASR + MT + TTS was previously used to power many commercial STST products, including Google Translate. It’s also a very data and compute efficient way of developing a STST system, since existing speech recognition and text-to-speech systems can be coupled together to yield a new STST model without any additional training.

## Speech translation
We’ll use the Whisper model for our speech translation system, since it’s capable of translating from over 96 languages to English. Specifically, we’ll load the Whisper Base checkpoint, which clocks in at 74M parameters. It’s by no means the most performant Whisper model, with the largest Whisper checkpoint being over 20x larger, but since we’re concatenating two auto-regressive systems together (ST + TTS), we want to ensure each model can generate relatively quickly so that we get reasonable inference speed:


# Text-to-speech
The second half of our cascaded STST system involves mapping from English text to English speech. For this, we’ll use the pre-trained SpeechT5 TTS model for English TTS. 🤗 Transformers currently doesn’t have a TTS pipeline, so we’ll have to use the model directly ourselves