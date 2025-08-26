import soundfile as sf

from salt.tokenization.audio_tokenizer import AudioTokenizer


def verify_audio_reconstruction(quantizer: AudioTokenizer, audio_dict: dict):
    codes = quantizer.encode(audio_dict)
    reconstructed = quantizer.decode(codes)
    reconstructed = reconstructed.detach().cpu().numpy()

    sf.write("reconstructed.wav", reconstructed.ravel(), quantizer.sample_rate)
