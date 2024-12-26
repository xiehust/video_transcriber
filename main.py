import os
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
import tempfile
import argparse
from pathlib import Path
import string
import random
from aws_transcribe import TranscribeTool
import json
import re
import logging
import boto3
import base64
import io
from pydub import AudioSegment
from extract_video_frames import extract_video_frames
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()

# Get SageMaker endpoint name from environment
endpoint_name = os.getenv('SAGEMAKER_ENDPOINT_NAME')
if not endpoint_name:
    logger.warning("SAGEMAKER_ENDPOINT_NAME not set in environment variables")

def random_string_name(length=12):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

class VideoTranscriber:
    def __init__(self, video_path, service="aws", language="Chinese"):
        """
        Initialize VideoTranscriber
        :param video_path: Path to video file
        :param service: Transcription service to use ('aws')
        :param language: Language of the audio
        """
        self.video_path = video_path
        self.video = VideoFileClip(video_path)
        self.service = service
        self.language = language
        
        if service == "aws":
            self.aws_transcribe = TranscribeTool()
            
    def extract_audio(self):
        """Extract audio from video and save it temporarily"""
        temp_file_name = random_string_name()
        os.makedirs("audio", exist_ok=True)
        temp_audio_path = os.path.join("audio", f"{temp_file_name}.mp3")
        
        # Extract audio from video
        self.video.audio.write_audiofile(temp_audio_path)
        return temp_audio_path

    def parse_aws_timestamp(self, line):
        """Parse AWS transcription timestamp line and extract start time, end time, and text"""
        try:
            # Extract the content between square brackets
            bracket_content = re.search(r'\[(.*?)\]', line)
            if not bracket_content:
                return None

            # Split the content after the closing bracket
            text = line[line.index(']')+1:].strip()
            
            # Parse the timestamp part
            timestamp_parts = bracket_content.group(1).split()
            
            # Extract the times (assuming format is either "1.46s-1.86s" or "spk_0 1.46s-1.86s")
            time_part = next(part for part in timestamp_parts if '-' in part)
            start_str, end_str = time_part.split('-')
            
            # Convert to float, removing 's' if present
            start_time = float(start_str.replace('s', ''))
            end_time = float(end_str.replace('s', ''))
            
            return {
                "start": start_time,
                "end": end_time,
                "text": text
            }
        except Exception as e:
            logger.error(f"Warning: Could not parse timestamp line: {line}")
            logger.error(f"Error: {e}")
            return None

    def transcribe_with_aws(self, audio_path):
        """Transcribe audio using AWS Transcribe"""
        try:
            # Get AWS credentials from environment variables
            aws_region = os.getenv('AWS_REGION', 'us-east-1')
            s3_bucket = os.getenv('AWS_S3_BUCKET')
            
            if not s3_bucket:
                raise ValueError("AWS_S3_BUCKET environment variable is required for AWS Transcribe")

            # Prepare parameters for AWS Transcribe
            params = {
                'file_url': audio_path,  # Pass the local file path directly
                'file_type': 'mp3',
                's3_bucket_name': s3_bucket,
                'aws_region': aws_region,
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 2
            }

            # If language is specified and not "auto", add it to parameters
            if self.language.lower() != "auto":
                # Map common language codes to AWS language codes
                language_map = {
                    "chinese": "zh-CN",
                    "english": "en-US",
                    "japanese": "ja-JP",
                    # Add more language mappings as needed
                }
                aws_language = language_map.get(self.language.lower())
                if aws_language:
                    params['language_code'] = aws_language
                    params['identify_language'] = False
                else:
                    params['identify_language'] = True

            # Call AWS Transcribe
            result = self.aws_transcribe.invoke(params)
            
            # Parse the result text which contains timestamps
            segments = []
            
            for line in result.split('\n'):
                if line.strip() and '[' in line and ']' in line:
                    segment = self.parse_aws_timestamp(line)
                    if segment:
                        segments.append(segment)
            
            return {"segments": segments} if segments else None
            
        except Exception as e:
            logger.error(f"Error during AWS transcription: {e}")
            return None

    def extract_video_segment(self, start_time, end_time, output_path):
        """Extract a segment of video between start_time and end_time"""
        try:
            # Get video duration and ensure end_time doesn't exceed it
            duration = self.video.duration
            end_time = min(end_time, duration)
            start_time = start_time if start_time > 0 else 0
            video_segment = self.video.subclip(start_time, end_time)
            video_segment.write_videofile(output_path)
            return True
        except Exception as e:
            logger.error(f"Error extracting video segment: {e}")
            return False

    def process_video(self, buffer, min_segment,output_dir="output"):
        """Process the video file and extract segments with speech"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extract audio
        yield "Extracting audio from video..."
        audio_path = self.extract_audio()
        
        # Transcribe audio using selected service
        yield f"Transcribing audio using {self.service}..."
        if self.service == "aws":
            transcript = self.transcribe_with_aws(audio_path)
        else:
            yield f"Error: Not supported service: {self.service}"
            return
        
        if not transcript:
            yield "Error: Transcription failed"
            return
        
        # Process segments
        segments = transcript.get("segments", [])
        total_segments = len(segments)
        yield f"Processing {total_segments} segments..."
        
        for idx, segment in enumerate(segments):
            start_time = segment.get("start") if isinstance(segment, dict) else segment.start
            end_time = segment.get("end") if isinstance(segment, dict) else segment.end
            text = segment.get("text") if isinstance(segment, dict) else segment.text

            if end_time - start_time < min_segment:
                continue
            
            yield f"Processing segment {idx + 1}/{total_segments}: {start_time:.2f}s - {end_time:.2f}s"
            
            # Create output filename
            output_filename = f"segment_{idx:03d}_{start_time:.2f}_{end_time:.2f}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            yield f"Extracting segment {idx + 1}: {text}"
            success = self.extract_video_segment(start_time-buffer, end_time + buffer , output_path)
            
            if success:
                yield f"Saved segment to: {output_path}"
                
                try:                    
                    # Save transcription text
                    text_path = output_path.replace('.mp4', '.txt')
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(f"{start_time}-{end_time} {text}\n")
                    
                except Exception as e:
                    yield f"Error: Failed to save transcription: {e}"
                    # Save original transcription as fallback
                    text_path = output_path.replace('.mp4', '.txt')
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(f"{start_time}-{end_time}: {text}")
            else:
                yield f"Error: Failed to extract segment {idx + 1}"

        # Clean up audio file
        try:
            os.remove(audio_path)
        except Exception as e:
            yield f"Warning: Could not remove temporary audio file: {e}"

    def __del__(self):
        """Clean up video file handle"""
        if hasattr(self, 'video'):
            self.video.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Enter the path to your video file")
    parser.add_argument("--service", choices=[ "aws"], default="aws",
                      help="Choose transcription service (aws)")
    parser.add_argument("--language", default="Chinese",
                      help="Language of the audio (e.g., Chinese, English, auto)")
    parser.add_argument("--buffer", default=1,type=float,
                      help="buffer for segment, default 1s")
    parser.add_argument("--minsegment", default=0.3,type=float,
                      help="minimum segment length default 0.3s")
    parser.add_argument("--method", default="uniform", choices=['uniform','random','difference'],
                      help="frame extraction method,default uniform")
    parser.add_argument("--num_frames", default=3,type=int,
                      help="number of frames to extract, default 3")
    parser.add_argument("--threshold", default=0.95,type=float,
                      help="threshold for frame extraction, default 0.95")
    parser.add_argument("--min_frame_diff", default=0.5,type=float,
                      help="minimum frame interval for frame extraction, default 0.5")
    parser.add_argument("--output_base_dir", default="./output",
                      help="output base directory, default ./output")

    args = parser.parse_args()
    video_path = args.file
    if not os.path.exists(video_path):
        logger.error("Error: Video file does not exist")
        return

    transcriber = VideoTranscriber(video_path, service=args.service, language=args.language)
    transcriber.process_video(buffer=args.buffer,
                                min_segment=args.minsegment,
                                output_dir=os.path.join(args.output_base_dir,os.path.basename(video_path).split('.')[0]))
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    extract_video_frames(os.path.join(args.output_base_dir,os.path.basename(video_path).split('.')[0]),
                         output_base_dir=os.path.join(args.output_base_dir,timestamp, os.path.basename(video_path).split('.')[0],'frames'),
                         method=args.method,
                         num_frames=args.num_frames,
                         threshold=args.threshold,
                         min_frame_diff=args.min_frame_diff)
    

if __name__ == "__main__":
    main()
