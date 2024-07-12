# Automatic Speech Recognition
 Automatic Speech Recognition (ASR) refers to a technology that allows humans to communicate with a computer interface using their voice in a manner similar to actual human conversations

# Pre-trained ASR Models
 ### 1. CTC model
  A CTC model is essentially an ‘acoustic-only’ model: it consists of an encoder which forms hidden-state representations from the audio inputs, and a linear layer which maps the hidden-states to characters:

  This means that the system almost entirely bases its prediction on the acoustic input it was given (the phonetic sounds of the audio), and so has a tendency to transcribe the audio in a phonetic way (e.g. CHRISTMAUS). It gives less importance to the language modelling context of previous and successive letters, and so is prone to phonetic spelling errors. A more intelligent model would identify that CHRISTMAUS is not a valid word in the English vocabulary, and correct it to CHRISTMAS when making its predictions. We’re also missing two big features in our prediction - casing and punctuation - which limits the usefulness of the model’s transcriptions to real-world applications
 # 2. Seq2Seq
Seq2Seq models are formed of an encoder and decoder linked via a cross-attention mechanism. The encoder plays the same role as before, computing hidden-state representations of the audio inputs, while the decoder plays the role of a language model. The decoder processes the entire sequence of hidden-state representations from the encoder and generates the corresponding text transcriptions. With global context of the audio input, the decoder is able to use language modelling context as it makes its predictions, correcting for spelling mistakes on-the-fly and thus circumventing the issue of phonetic predictions.

There are two downsides to Seq2Seq models:

1. They are inherently slower at decoding, since the decoding process happens one step at a time, rather than all at once
2. They are more data hungry, requiring significantly more training data to reach convergence

In particular, the need for large amounts of training data has been a bottleneck in the advancement of Seq2Seq architectures for speech. Labelled speech data is difficult to come by, with the largest annotated datasets at the time clocking in at just 10,000 hours. This all changed in 2022 upon the release of Whisper. Whisper is a pre-trained model for speech recognition published in September 2022 by the authors Alec Radford et al. from OpenAI. Unlike its CTC predecessors, which were pre-trained entirely on un-labelled audio data, Whisper is pre-trained on a vast quantity of labelled audio-transcription data, 680,000 hours to be precise.

# Fine-tuning Whisper for ASR
fine-tune the Whisper model for the Dhivehi language. However, the steps covered here generalise to any language in the Common Voice dataset, and more generally to any ASR dataset on the Hugging Face Hub.
## Dataset
[Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) contains approximately ten hours of labelled Dhivehi data, three of which is held-out test data. This is extremely little data for fine-tuning, so we’ll be relying on leveraging the extensive multilingual ASR knowledge acquired by Whisper during pre-training for the low-resource Dhivehi language.

## Feature Extractor, Tokenizer and Processor
The ASR pipeline can be de-composed into three stages:

1. The feature extractor which pre-processes the raw audio-inputs to log-mel spectrograms
2. The model which performs the sequence-to-sequence mapping
3. The tokenizer which post-processes the predicted tokens to text

In huggingface Transformers, the Whisper model has an associated feature extractor and tokenizer, called WhisperFeatureExtractor and WhisperTokenizer respectively. To make our lives simple, these two objects are wrapped under a single class, called the WhisperProcessor. We can call the WhisperProcessor to perform both the audio pre-processing and the text token post-processing. In doing so, we only need to keep track of two objects during training: the processor and the model.

When performing multilingual fine-tuning, we need to set the "language" and "task" when instantiating the processor. The "language" should be set to the source audio language, and the task to "transcribe" for speech recognition or "translate" for speech translation. These arguments modify the behaviour of the tokenizer, and should be set correctly to ensure the target labels are encoded properly.


# Training and Evaluation
Now that we’ve prepared our data, we’re ready to dive into the training pipeline. The 🤗 Trainer will do much of the heavy lifting for us. All we have to do is:

* Define a data collator: the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.

* Evaluation metrics: during evaluation, we want to evaluate the model using the word error rate (WER) metric. We need to define a compute_metrics function that handles this computation.

* Load a pre-trained checkpoint: we need to load a pre-trained checkpoint and configure it correctly for training.

* Define the training arguments: these will be used by the 🤗 Trainer in constructing the training schedule.

Once we’ve fine-tuned the model, we will evaluate it on the test data to verify that we have correctly trained it to transcribe speech in Dhivehi.

## Define a Data Collator
The data collator for a sequence-to-sequence speech model is unique in the sense that it treats the input_features and labels independently: the input_features must be handled by the feature extractor and the labels by the tokenizer.

The input_features are already padded to 30s and converted to a log-Mel spectrogram of fixed dimension, so all we have to do is convert them to batched PyTorch tensors. We do this using the feature extractor’s .pad method with return_tensors=pt. Note that no additional padding is applied here since the inputs are of fixed dimension, the input_features are simply converted to PyTorch tensors.

On the other hand, the labels are un-padded. We first pad the sequences to the maximum length in the batch using the tokenizer’s .pad method. The padding tokens are then replaced by -100 so that these tokens are not taken into account when computing the loss. We then cut the start of transcript token from the beginning of the label sequence as we append it later during training.

We can leverage the WhisperProcessor we defined earlier to perform both the feature extractor and the tokenizer operations: