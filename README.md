# Vikhr Salt: Speech And Language Transformer

![Vikhr Salt Logo](https://huggingface.co/Vikhrmodels/salt-116k/resolve/main/IMG_1304%20copy.png)

Vikhr Salt is a multimodal model based on a pre-trained large language model, extended with new audio tokens to handle both TTS (text-to-speech) and ASR (automatic speech recognition) tasks. The model incorporates two variants for encoding audio—Encodec and SpeechTokenizer—and achieves stable training by fine-tuning precision settings. This approach allows Vikhr Salt to leverage pre-existing LLM knowledge while effectively generating and understanding speech, marking a step forward in multimodal learning.

## Install requirements 
Clone audio tokenizers' repositories: 

```
git clone https://github.com/jishengpeng/WavTokenizer.git
git clone https://github.com/Aria-K-Alethia/BigCodec.git

echo "/path/to/Salt/WavTokenizer" > "$(poetry env info --path)/lib/pythonX.Y/site-packages/wavtokenizer.pth"
echo "/path/to/Salt/BigCodec" > "$(poetry env info --path)/lib/pythonX.Y/site-packages/bigcodec.pth"

```
Setup environment: 
```
pip install poetry
poetry install

```
Download audio tokenizers checkpoints: 
[BigCodec](https://github.com/Aria-K-Alethia/BigCodec/tree/main)
[WavTokenizer](https://huggingface.co/novateur/WavTokenizer-large-unify-40token)
[SpeechTokenizer](https://huggingface.co/fnlp/SpeechTokenizer/tree/main)

## How to run
### Preparing Data
Specify dataset and tokenizer in configs/tokenization and run
```
python prepare_data.py --config_name default.yaml --config_path configs 

```

### Training
Specify training configuration in configs/default.yaml

for single gpu
```
source scripts/run_me.sh

```

for multi gpu+ds2
```
source scripts/run_me_ds2.sh

```

## Customization
### Tokenizers 
BigCodec, WavTokenizer, and SpeechTokenizer are implemented. 
To add new tokenizer, create new class inheriting from [AudioTokenizer](salt/tokenization/audio_tokenizer.py).

### Datasets
Supported datasets formats: text only, audio only, tts, asr, instruct (asr/tts). 
To add new dataset, implement new class inheriting from [SaltDataset](salt/data/base_datasets.py). 
