import io
import unittest
import numpy as np
import soundfile as sf
from normalization import normalize_ar
from audio_utils import audio_example_to_waveform

class UtilitiesTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_ar("إنتَ عامل إيه؟"), "انت عامل ايه؟")
        self.assertEqual(normalize_ar("الســــلام ٥!"), "السلام 5")
    def test_audio_bytes_preferred(self):
        b = io.BytesIO()
        sf.write(b, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
        wav, sr = audio_example_to_waveform({"bytes": b.getvalue(), "path": "missing.wav"})
        self.assertEqual(sr, 16000)
        self.assertEqual(wav.shape, (1600,))

if __name__ == "__main__":
    unittest.main()
