import librosa
import numpy as np
import io
from pydub import AudioSegment

def analyze_audio(file_bytes, sample_rate=22050):
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_channels(1).set_frame_rate(sample_rate)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / np.max(np.abs(samples))

    stft = np.abs(librosa.stft(samples, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sample_rate)

    energy_per_freq = np.sum(stft, axis=1)
    p = energy_per_freq / (np.sum(energy_per_freq) + 1e-9)
    spectral_entropy = -np.sum(p * np.log2(p + 1e-9))

    low_mask = freqs < 250
    mid_mask = (freqs >= 250) & (freqs < 4000)
    high_mask = freqs >= 4000

    band_energy = {
        'low': np.sum(stft[low_mask]) / (np.sum(stft) + 1e-9),
        'mid': np.sum(stft[mid_mask]) / (np.sum(stft) + 1e-9),
        'high': np.sum(stft[high_mask]) / (np.sum(stft) + 1e-9)
    }

    onset_env = librosa.onset.onset_strength(y=samples, sr=sample_rate)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate)
    onset_density = len(onsets) / (len(samples) / sample_rate)

    rms = librosa.feature.rms(y=samples)[0]
    dynamic_range = np.max(rms) - np.min(rms)

    instruments = {
        'drums': min(1.0, onset_density / 8.0),
        'bass': min(1.0, band_energy['low'] * 3.0),
        'melody': min(1.0, band_energy['mid'] * 2.5),
        'vocals': min(1.0, (band_energy['mid'] * 0.8 + band_energy['high'] * 1.2))
    }

    genre_profiles = {
        'synthwave': {'spectral_entropy': 6.0, 'onset_density': 3.5, 'band_energy': {'low':0.4, 'mid':0.4, 'high':0.2}},
        'pop':       {'spectral_entropy': 5.5, 'onset_density': 4.0, 'band_energy': {'low':0.3, 'mid':0.5, 'high':0.2}},
        'techno':    {'spectral_entropy': 7.0, 'onset_density': 6.0, 'band_energy': {'low':0.6, 'mid':0.3, 'high':0.1}},
        'lo-fi':     {'spectral_entropy': 4.0, 'onset_density': 2.0, 'band_energy': {'low':0.5, 'mid':0.4, 'high':0.1}},
        'jazz':      {'spectral_entropy': 5.0, 'onset_density': 3.0, 'band_energy': {'low':0.2, 'mid':0.6, 'high':0.2}}
    }

    genre_estimates = {}
    for g, prof in genre_profiles.items():
        dist = (
            (spectral_entropy - prof['spectral_entropy'])**2 +
            (onset_density - prof['onset_density'])**2 +
            (band_energy['low'] - prof['band_energy']['low'])**2 +
            (band_energy['mid'] - prof['band_energy']['mid'])**2 +
            (band_energy['high'] - prof['band_energy']['high'])**2
        ) ** 0.5
        genre_estimates[g] = 1.0 / (1.0 + dist)

    total = sum(genre_estimates.values())
    if total > 0:
        for g in genre_estimates:
            genre_estimates[g] /= total

    return {
        'spectral_entropy': float(spectral_entropy),
        'band_energy': {k: float(v) for k, v in band_energy.items()},
        'onset_density': float(onset_density),
        'dynamic_range': float(dynamic_range),
        'instruments': instruments,
        'genre_estimates': genre_estimates
    }