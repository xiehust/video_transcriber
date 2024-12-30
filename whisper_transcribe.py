import whisper
import torch
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
import os
import ssl
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperTranscribeTool:
    """Tool for transcribing audio using OpenAI's Whisper model."""
    
    SUPPORTED_MODEL_SIZES = ['tiny', 'base', 'small', 'medium', 'large']
    
    def __init__(self, model_size: str = "base", disable_ssl_verification: bool = False):
        """
        Initialize WhisperTranscribeTool.
        
        Args:
            model_size: Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
            disable_ssl_verification: Whether to disable SSL verification when downloading model
        """
        if model_size not in self.SUPPORTED_MODEL_SIZES:
            raise ValueError(f"Model size must be one of {self.SUPPORTED_MODEL_SIZES}")
            
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.disable_ssl_verification = disable_ssl_verification
        
    def load_model(self) -> None:
        """Load Whisper model with SSL verification handling."""
        if self.model is None:
            logger.info(f"Loading Whisper {self.model_size} model on {self.device}...")
            try:
                if self.disable_ssl_verification:
                    # Temporarily disable SSL verification
                    original_context = ssl._create_default_https_context
                    ssl._create_default_https_context = ssl._create_unverified_context
                
                self.model = whisper.load_model(self.model_size, device=self.device)
                
                if self.disable_ssl_verification:
                    # Restore original SSL context
                    ssl._create_default_https_context = original_context
                    
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                if not self.disable_ssl_verification:
                    logger.info("Retrying with SSL verification disabled...")
                    self.disable_ssl_verification = True
                    return self.load_model()
                raise
            
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code for transcription
            
        Returns:
            Dictionary containing transcription results
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If transcription fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        self.load_model()
        
        try:
            logger.info(f"Transcribing {audio_path}...")
            result = self.model.transcribe(
                audio_path,
                language=language,
                verbose=False
            )
            
            # Format segments with timestamps
            segments = []
            for segment in result['segments']:
                segments.append({
                    "start": segment['start'],
                    "end": segment['end'],
                    "text": segment['text'].strip()
                })
                
            return {
                "text": result['text'].strip(),
                "segments": segments,
                "language": result['language']
            }
            
        except Exception as e:
            logger.error(f"Error during Whisper transcription: {e}")
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def invoke(self, tool_parameters: Dict[str, Any]) -> str:
        """
        Invoke Whisper transcription.
        
        Args:
            tool_parameters: Dictionary containing:
                - audio_path: Path to audio file (required)
                - language: Language code (optional)
                
        Returns:
            JSON string containing transcription results or error message
        """
        try:
            audio_path = tool_parameters.get('audio_path')
            language = tool_parameters.get('language')
            
            if not audio_path:
                return json.dumps({
                    "error": "audio_path parameter is required"
                })
            
            if not os.path.exists(audio_path):
                return json.dumps({
                    "error": f"Audio file not found: {audio_path}"
                })
                
            result = self.transcribe_audio(audio_path, language)
            return json.dumps({
                "success": True,
                "result": result
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
