import unittest
import os
import tempfile
from whisper_transcribe import WhisperTranscribeTool
import json

class TestWhisperTranscribeTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary audio file for testing
        cls.test_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        cls.test_audio.write(b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xbb\x00\x00\x00\xee\x02\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
        cls.test_audio.close()

    @classmethod
    def tearDownClass(cls):
        # Clean up the temporary file
        os.unlink(cls.test_audio.name)

    def test_initialization(self):
        # Test initialization with valid model sizes
        for model_size in WhisperTranscribeTool.SUPPORTED_MODEL_SIZES:
            tool = WhisperTranscribeTool(model_size=model_size)
            self.assertEqual(tool.model_size, model_size)
            self.assertIsNone(tool.model)
            self.assertFalse(tool.disable_ssl_verification)

        # Test initialization with invalid model size
        with self.assertRaises(ValueError):
            WhisperTranscribeTool(model_size="invalid_size")

        # Test initialization with SSL verification disabled
        tool = WhisperTranscribeTool(disable_ssl_verification=True)
        self.assertTrue(tool.disable_ssl_verification)

    def test_transcribe_valid_audio(self):
        tool = WhisperTranscribeTool(model_size="tiny")  # Use tiny model for faster testing
        result = tool.invoke({'audio_path': self.test_audio.name})
        
        # Verify the result is valid JSON
        try:
            json_result = json.loads(result)
            self.assertIsInstance(json_result, dict)
            self.assertTrue(json_result.get('success', False))
            
            result_data = json_result.get('result', {})
            self.assertIn('text', result_data)
            self.assertIn('segments', result_data)
            self.assertIn('language', result_data)
            
            # Verify text is stripped
            self.assertEqual(result_data['text'], result_data['text'].strip())
            
            # Verify segments format
            for segment in result_data['segments']:
                self.assertIn('start', segment)
                self.assertIn('end', segment)
                self.assertIn('text', segment)
                self.assertEqual(segment['text'], segment['text'].strip())
                
        except json.JSONDecodeError:
            self.fail("Result is not valid JSON")

    def test_missing_audio_file(self):
        tool = WhisperTranscribeTool()
        result = tool.invoke({'audio_path': 'nonexistent.wav'})
        json_result = json.loads(result)
        self.assertIn('error', json_result)
        self.assertIn('not found', json_result['error'])
        self.assertFalse(json_result.get('success', True))

    def test_missing_audio_path(self):
        tool = WhisperTranscribeTool()
        result = tool.invoke({})
        json_result = json.loads(result)
        self.assertIn('error', json_result)
        self.assertIn('audio_path parameter is required', json_result['error'])
        self.assertFalse(json_result.get('success', True))

    def test_language_specification(self):
        tool = WhisperTranscribeTool(model_size="tiny")  # Use tiny model for faster testing
        result = tool.invoke({
            'audio_path': self.test_audio.name,
            'language': 'Chinese'
        })
        json_result = json.loads(result)
        self.assertIsInstance(json_result, dict)
        self.assertTrue(json_result.get('success', False))
        
        result_data = json_result.get('result', {})
        self.assertIn('text', result_data)
        self.assertIn('segments', result_data)
        self.assertIn('language', result_data)

    def test_ssl_verification(self):
        # Test with SSL verification disabled
        tool = WhisperTranscribeTool(disable_ssl_verification=True)
        result = tool.invoke({'audio_path': self.test_audio.name})
        json_result = json.loads(result)
        self.assertTrue(json_result.get('success', False))

if __name__ == '__main__':
    unittest.main()
