# Audio Classification
 Audio classification - just like with text - assigns a class label output from the input data. The only difference is instead of text inputs, you have raw audio waveforms. Some practical applications of audio classification include identifying speaker intent, language classification, and even animal species by their sounds.

 ## Pre_trained models and datasets for audio classification
 A berif introduction about Audio classifiction models and datasets.
 ## Fine-tuning a model for music classification
 **Genre music classification** is an important task that classify musics for better recommndation
 ### Dataset
 To train our model, we’ll use the [GTZAN](https://huggingface.co/datasets/google/speech_commands) dataset, which is a popular dataset of 1,000 songs for music genre classification. Each song is a 30-second clip from one of 10 genres of music, spanning disco to metal. We can get the audio files and their corresponding labels from the Hugging Face Hub with the load_dataset() function from 🤗 Datasets:

### Picking a pretrained model for audio classification

To get started, let’s pick a suitable pretrained model for audio classification. In this domain, pretraining is typically carried out on large amounts of unlabeled audio data, using datasets like LibriSpeech and Voxpopuli. Although models like Wav2Vec2 and HuBERT are very popular, we’ll use a model called DistilHuBERT. This is a much smaller (or distilled) version of the HuBERT model, which trains around 73% faster, yet preserves most of the performance.

## Results
The model accuracy at epoch 7 is **85%**
  

