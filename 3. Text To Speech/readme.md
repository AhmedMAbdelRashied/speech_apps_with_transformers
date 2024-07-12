# Text To Speech
Text to speech (TTS) is a technology that converts text into spoken audio. It can read aloud PDFs, websites, and books using natural AI voices.

# Pre-trained models for TTS applications
## SpeechT5

SpeechT5 is a model published by Junyi Ao et al. from Microsoft that is capable of handling a range of speech tasks. While in this unit, we focus on the text-to-speech aspect, this model can be tailored to speech-to-text tasks (automatic speech recognition or speaker identification), as well as speech-to-speech (e.g. speech enhancement or converting between different voices). This is due to how the model is designed and pre-trained.

At the heart of SpeechT5 is a regular Transformer encoder-decoder model. Just like any other Transformer, the encoder-decoder network models a sequence-to-sequence transformation using hidden representations. This Transformer backbone is the same for all tasks SpeechT5 supports.

This Transformer is complemented with six modal-specific (speech/text) pre-nets and post-nets. The input speech or text (depending on the task) is preprocessed through a corresponding pre-net to obtain the hidden representations that Transformer can use. The Transformer’s output is then passed to a post-net that will use it to generate the output in the target modality.

This is what the architecture looks like (image from the original paper):
<img src='https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/speecht5/architecture.jpg'>

SpeechT5 is first pre-trained using large-scale unlabeled speech and text data, to acquire a unified representation of different modalities. During the pre-training phase all pre-nets and post-nets are used simultaneously.

After pre-training, the entire encoder-decoder backbone is fine-tuned for each individual task. At this step, only the pre-nets and post-nets relevant to the specific task are employed. For example, to use SpeechT5 for text-to-speech, you’d need the text encoder pre-net for the text inputs and the speech decoder pre- and post-nets for the speech outputs.

This approach allows to obtain several models fine-tuned for different speech tasks that all benefit from the initial pre-training on unlabeled data.

Let’s see what are the pre- and post-nets that SpeechT5 uses for the TTS task specifically:

* Text encoder pre-net: A text embedding layer that maps text tokens to the hidden representations that the encoder expects. This is similar to what happens in an NLP model such as BERT.
* Speech decoder pre-net: This takes a log mel spectrogram as input and uses a sequence of linear layers to compress the spectrogram into hidden representations.
* Speech decoder post-net: This predicts a residual to add to the output spectrogram and is used to refine the results.

When combined, this is what SpeechT5 architecture for text-to-speech looks like:

<img src='https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/speecht5/tts.jpg'>


# Bark
Bark is a transformer-based text-to-speech model proposed by Suno AI in suno-ai/bark.

Unlike SpeechT5, Bark generates raw speech waveforms directly, eliminating the need for a separate vocoder during inference – it’s already integrated. This efficiency is achieved through the utilization of Encodec, which serves as both a codec and a compression tool.

With Encodec, you can compress audio into a lightweight format to reduce memory usage and subsequently decompress it to restore the original audio. This compression process is facilitated by 8 codebooks, each consisting of integer vectors. Think of these codebooks as representations or embeddings of the audio in integer form. It’s important to note that each successive codebook improves the quality of the audio reconstruction from the previous codebooks. As codebooks are integer vectors, they can be learned by transformer models, which are very efficient in this task. This is what Bark was specifically trained to do.

To be more specific, Bark is made of 4 main models:

1. BarkSemanticModel (also referred to as the ‘text’ model): a causal auto-regressive transformer model that takes as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.
2. BarkCoarseModel (also referred to as the ‘coarse acoustics’ model): a causal autoregressive transformer, that takes as input the results of the BarkSemanticModel model. It aims at predicting the first two audio codebooks necessary for EnCodec.
3. BarkFineModel (the ‘fine acoustics’ model), this time a non-causal autoencoder transformer, which iteratively predicts the last codebooks based on the sum of the previous codebooks embeddings.
4. having predicted all the codebook channels from the EncodecModel, Bark uses it to decode the output audio array.

## Massive Multilingual Speech (MMS)
What if you are looking for a pre-trained model in a language other than English? Massive Multilingual Speech (MMS) is another model that covers an array of speech tasks, however, it supports a large number of languages. For instance, it can synthesize speech in over 1,100 languages.

MMS for text-to-speech is based on VITS Kim et al., 2021, which is one of the state-of-the-art TTS approaches.

VITS is a speech generation network that converts text into raw speech waveforms. It works like a conditional variational auto-encoder, estimating audio features from the input text. First, acoustic features, represented as spectrograms, are generated. The waveform is then decoded using transposed convolutional layers adapted from HiFi-GAN. During inference, the text encodings are upsampled and transformed into waveforms using the flow module and HiFi-GAN decoder. Like Bark, there’s no need for a vocoder, as waveforms are generated directly.