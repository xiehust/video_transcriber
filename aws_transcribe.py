import json
import logging
from typing import Any, Union,List
import uuid
import boto3
from botocore.exceptions import BotoCoreError
from pydantic import BaseModel, Field, field_validator
import time
import requests
from urllib.parse import urlparse
import os
from botocore.exceptions import ClientError
from requests.exceptions import RequestException
import re
from transcript_process import TranscriptProcessor, PRO_MODEL_ID, LITE_MODEL_ID, CLAUDE_SONNET_35_MODEL_ID

transcipt_sentence = TranscriptProcessor(model_id=PRO_MODEL_ID)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LanguageCodeOptions = [
    'af-ZA', 'ar-AE', 'ar-SA', 'da-DK', 'de-CH', 'de-DE', 'en-AB', 'en-AU', 'en-GB', 'en-IE',
    'en-IN', 'en-US', 'en-WL', 'es-ES', 'es-US', 'fa-IR', 'fr-CA', 'fr-FR', 'he-IL', 'hi-IN',
    'id-ID', 'it-IT', 'ja-JP', 'ko-KR', 'ms-MY', 'nl-NL', 'pt-BR', 'pt-PT', 'ru-RU', 'ta-IN',
    'te-IN', 'tr-TR', 'zh-CN', 'zh-TW', 'th-TH', 'en-ZA', 'en-NZ', 'vi-VN', 'sv-SE', 'ab-GE',
    'ast-ES', 'az-AZ', 'ba-RU', 'be-BY', 'bg-BG', 'bn-IN', 'bs-BA', 'ca-ES', 'ckb-IQ', 'ckb-IR',
    'cs-CZ', 'cy-WL', 'el-GR', 'et-ET', 'eu-ES', 'fi-FI', 'gl-ES', 'gu-IN', 'ha-NG', 'hr-HR',
    'hu-HU', 'hy-AM', 'is-IS', 'ka-GE', 'kab-DZ', 'kk-KZ', 'kn-IN', 'ky-KG', 'lg-IN', 'lt-LT',
    'lv-LV', 'mhr-RU', 'mi-NZ', 'mk-MK', 'ml-IN', 'mn-MN', 'mr-IN', 'mt-MT', 'no-NO', 'or-IN',
    'pa-IN', 'pl-PL', 'ps-AF', 'ro-RO', 'rw-RW', 'si-LK', 'sk-SK', 'sl-SI', 'so-SO', 'sr-RS',
    'su-ID', 'sw-BI', 'sw-KE', 'sw-RW', 'sw-TZ', 'sw-UG', 'tl-PH', 'tt-RU', 'ug-CN', 'uk-UA',
    'uz-UZ', 'wo-SN', 'zu-ZA'
]

MediaFormat = ['mp3', 'mp4', 'wav', 'flac', 'ogg', 'amr', 'webm', 'm4a']

def is_url(text):
    if not text:
        return False
    text = text.strip()
    # Regular expression pattern for URL validation
    pattern = re.compile(
        r'^'  # Start of the string
        r'(?:http|https)://'  # Protocol (http or https)
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # Domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP address
        r'(?::\d+)?'  # Optional port
        r'(?:/?|[/?]\S+)'  # Path
        r'$',  # End of the string
        re.IGNORECASE
    )
    return bool(pattern.match(text))

def upload_to_s3(s3_client, file_path_or_url, bucket_name, s3_key=None, max_retries=3):
    """
    Upload a file to S3 from either a local path or URL
    
    Parameters:
    - s3_client: boto3 S3 client
    - file_path_or_url (str): Local file path or URL of the file to upload
    - bucket_name (str): The name of the S3 bucket
    - s3_key (str): The desired key (path) in S3. If None, will use the filename
    - max_retries (int): Maximum number of retry attempts
    
    Returns:
    - tuple: (str, str) - (S3 URI if successful, error message if failed)
    """
    
    if not file_path_or_url or not bucket_name:
        return None, "File path/URL and bucket name are required"

    retry_count = 0
    while retry_count < max_retries:
        try:
            # If s3_key is not provided, generate one from the file path/URL
            if not s3_key:
                if is_url(file_path_or_url):
                    parsed_url = urlparse(file_path_or_url)
                    filename = os.path.basename(parsed_url.path)
                else:
                    filename = os.path.basename(file_path_or_url)
                s3_key = f'transcribe-files/{filename}'

            # Handle URL vs local file
            if is_url(file_path_or_url):
                # Download from URL and upload to S3
                response = requests.get(file_path_or_url, stream=True, timeout=30)
                response.raise_for_status()
                s3_client.upload_fileobj(
                    response.raw,
                    bucket_name,
                    s3_key,
                    ExtraArgs={
                        'ContentType': response.headers.get('content-type'),
                        'ACL': 'private'
                    }
                )
            else:
                # Upload local file to S3
                if not os.path.exists(file_path_or_url):
                    return None, f"Local file not found: {file_path_or_url}"
                
                s3_client.upload_file(
                    file_path_or_url,
                    bucket_name,
                    s3_key,
                    ExtraArgs={'ACL': 'private'}
                )

            return f"s3://{bucket_name}/{s3_key}", f"Successfully uploaded file to s3://{bucket_name}/{s3_key}"

        except RequestException as e:
            retry_count += 1
            if retry_count == max_retries:
                return None, f"Failed to handle file after {max_retries} attempts: {str(e)}"
            continue

        except ClientError as e:
            return None, f"AWS S3 error: {str(e)}"

        except Exception as e:
            return None, f"Unexpected error: {str(e)}"

    return None, "Maximum retries exceeded"

class TranscribeTool():
    s3_client : Any = None
    transcribe_client: Any = None

    """
    Note that you must include one of LanguageCode, IdentifyLanguage,
    or IdentifyMultipleLanguages in your request. If you include more than one of these parameters, your transcription job fails.
    """
    def _transcribe_audio(self,
                          audio_file_uri,
                          file_type,
                          **extra_args
                          ):
        uuid_str = str(uuid.uuid4())
        job_name = f"{int(time.time())}-{uuid_str}"
        try:
            # Start transcription job
            response = self.transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': audio_file_uri},
                MediaFormat=file_type,  # Add MediaFormat parameter
                **extra_args
            )
            
            # Wait for the job to complete
            while True:
                status = self.transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(5)
                
            if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
                return status['TranscriptionJob']['Transcript']['TranscriptFileUri'], None
            else:
                return None, f"Error: TranscriptionJobStatus:{status['TranscriptionJob']['TranscriptionJobStatus']} "
                
        except Exception as e:
            return None, f"Error: {str(e)}"

    def _download_and_read_transcript(self, transcript_file_uri: str, max_retries: int = 3, sentences_mappings : str = '',ignore_unrelated:bool = True) -> tuple[str, str, float,float]:
        """
        Download and read the transcript file from the given URI.
        
        Parameters:
        - transcript_file_uri (str): The URI of the transcript file
        - max_retries (int): Maximum number of retry attempts
        
        Returns:
        - tuple: (text, error) - (Transcribed text if successful, error message if failed)
        """
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Download the transcript file
                response = requests.get(transcript_file_uri, timeout=300)
                response.raise_for_status()
                
                # Parse the JSON content
                transcript_data = response.json()

                # Check if speaker labels are present and enabled
                has_speaker_labels = ('results' in transcript_data and 
                                   'speaker_labels' in transcript_data['results'] and 
                                   'segments' in transcript_data['results']['speaker_labels'])

                logger.info(f"has_speaker_labels:{has_speaker_labels}")
                logger.info(f"transcript_data:{transcript_data}")
                
                if 'results' in transcript_data:
                    # Get speaker segments
                    segments = transcript_data['results']['audio_segments']
                    transcript_parts = []
                    for segment in segments:
                        start_time = segment['start_time']
                        end_time = segment['end_time']
                        transcript = segment['transcript']
                        transcript_processed = transcipt_sentence.process(transcript,sentences_mappings)
                 
                        if '无关内容' in transcript_processed :
                            transcript_processed += f" ({transcript})"
                            if ignore_unrelated:
                                continue
                
                        if has_speaker_labels:
                            speaker_label = segment['speaker_label']
                            timestamp_info = f"[{speaker_label} {start_time}s-{end_time}s]: {transcript_processed}"
                        else:
                            timestamp_info = f"[{start_time}s-{end_time}s]: {transcript_processed}"
                        transcript_parts.append(f"\n{timestamp_info}")
                    return ' '.join(transcript_parts).strip(), ''
                else:
                    return None, "No transcripts found in the response"
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count == max_retries:
                    return None, f"Failed to download transcript file after {max_retries} attempts: {str(e)}"
                continue
                
            except json.JSONDecodeError as e:
                return None, f"Failed to parse transcript JSON: {str(e)}"
                
            except Exception as e:
                return None, f"Unexpected error while processing transcript: {str(e)}"
                
        return None, "Maximum retries exceeded"
    
    def create_text_message(self, text: str):
        logger.info(text)
        return text
    
    def invoke(self, 
                tool_parameters: dict[str, Any], 
            ) :
            """
            Invoke AWS Transcribe tool
            """
            try:
                if not self.transcribe_client:
                    aws_region = tool_parameters.get('aws_region')
                    if aws_region:
                        self.transcribe_client = boto3.client("transcribe", region_name=aws_region)
                        self.s3_client = boto3.client("s3", region_name=aws_region)
                    else:
                        self.transcribe_client = boto3.client("transcribe")
                        self.s3_client = boto3.client("s3")

                file_url = tool_parameters.get('file_url')
                file_type = tool_parameters.get('file_type')
                language_code = tool_parameters.get('language_code')
                identify_language = tool_parameters.get('identify_language', True)
                identify_multiple_languages = tool_parameters.get('identify_multiple_languages', False)
                language_options_str = tool_parameters.get('language_options')
                s3_bucket_name = tool_parameters.get('s3_bucket_name')
                ShowSpeakerLabels = tool_parameters.get('ShowSpeakerLabels', True)
                MaxSpeakerLabels = tool_parameters.get('MaxSpeakerLabels', 1)
                sentences_mappings = tool_parameters.get('sentences_mappings', '')
                ignore_unrelated = tool_parameters.get('ignore_unrelated', False)

                # Check the input params
                if not s3_bucket_name:
                    return self.create_text_message(text=f"s3_bucket_name is required")
                
                # Validate file type
                if file_type not in MediaFormat:
                    return self.create_text_message(text=f"MediaFormat:{file_type} is not supported, should be one of {MediaFormat}")
                
                language_options = None
                if language_options_str:
                    language_options = language_options_str.split('|')
                    for lang in language_options:
                        if lang not in LanguageCodeOptions:
                            return self.create_text_message(text=f"{lang} is not supported, should be one of {LanguageCodeOptions}")
                
                if language_code and language_code not in LanguageCodeOptions:
                    return self.create_text_message(text=f"language_code:{language_code} is not supported, should be one of {LanguageCodeOptions}")
                
                if not language_code:
                    if identify_language and identify_multiple_languages:
                        return self.create_text_message(text=f"identify_language:{identify_language},identify_multiple_languages:{identify_multiple_languages}, Note that you must include one of LanguageCode, IdentifyLanguage,or IdentifyMultipleLanguages in your request. If you include more than one of these parameters, your transcription job fails.")
                else:
                    if identify_language or identify_multiple_languages:
                        return self.create_text_message(text=f"identify_language:{identify_language},identify_multiple_languages:{identify_multiple_languages}, Note that you must include one of LanguageCode, IdentifyLanguage,or IdentifyMultipleLanguages in your request. If you include more than one of these parameters, your transcription job fails.")

                extra_args = {
                    "IdentifyLanguage": identify_language,
                    "IdentifyMultipleLanguages": identify_multiple_languages
                }
                if language_code:
                    extra_args['LanguageCode'] = language_code
                if language_options:
                    extra_args['LanguageOptions'] = language_options
                if ShowSpeakerLabels:
                    extra_args['Settings'] = {"ShowSpeakerLabels": ShowSpeakerLabels, "MaxSpeakerLabels": MaxSpeakerLabels}
                
                if not file_url.startswith("s3://"):
                    # Upload to s3 bucket
                    s3_path_result, error = upload_to_s3(self.s3_client, file_url, s3_bucket_name)
                    if not s3_path_result:
                        return self.create_text_message(text=error)
                else:
                    s3_path_result = file_url

                transcript_file_uri, error = self._transcribe_audio(
                    audio_file_uri=s3_path_result,
                    file_type=file_type,
                    **extra_args
                )
                if not transcript_file_uri:
                    return self.create_text_message(text=error)

                # Download and read the transcript
                transcript_text, error = self._download_and_read_transcript(transcript_file_uri,sentences_mappings = sentences_mappings,ignore_unrelated=ignore_unrelated)
                if not transcript_text:
                    return self.create_text_message(text=error)

                return self.create_text_message(text=transcript_text)
                
            except Exception as e:
                return self.create_text_message(f'Exception {str(e)}')
