# Audio projects with Transformers
By nature, a sound wave is a continuous signal, meaning it contains an infinite number of signal values in a given time. This poses problems for digital devices which expect finite arrays. To be processed, stored, and transmitted by digital devices, the continuous sound wave needs to be converted into a series of discrete values, known as a digital representation.

If you look at any audio dataset, you’ll find digital files with sound excerpts, such as text narration or music. You may encounter different file formats such as .wav (Waveform Audio File), .flac (Free Lossless Audio Codec) and .mp3 (MPEG-1 Audio Layer 3). These formats mainly differ in how they compress the digital representation of the audio signal.

# Audio Basic Applications

## Audio classification:
easily categorize audio clips into different categories. You can identify whether a recording is of a barking dog or a meowing cat, or what music genre a song belongs to.

## Automatic speech recognition:
transform audio clips into text by transcribing them automatically. You can get a text representation of a recording of someone speaking, like “How are you doing today?“. Rather useful for note taking!
## Speaker diarization:
Ever wondered who’s speaking in a recording? With Transformers, you can identify which speaker is talking at any given time in an audio clip. Imagine being able to differentiate between “Alice” and “Bob” in a recording of them having a conversation.
## Text to speech:
create a narrated version of a text that can be used to produce an audio book, help with accessibility, or give a voice to an NPC in a game. With 🤗 Transformers, you can easily do that!