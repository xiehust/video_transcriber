# Video Transcriber

This tool transcribes speech from video files and extracts video segments containing speech. It supports AWS Transcribe services for transcription.

## Features

- Extract audio from video files
- Transcribe speech using AWS Transcribe
- Extract video segments containing speech
- Save transcription text alongside video segments
- Extract keyframes
- Classify the keyframe images
- Support for multiple languages

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` with your credentials:
- For OpenAI: Add your OpenAI API key
- For AWS Transcribe: Add your AWS credentials and S3 bucket name

## Usage

Basic usage:
```bash
python webui.py
```


### Language Support

- For AWS Transcribe, common languages are mapped to AWS language codes (e.g., "Chinese" -> "zh-CN")
- Use "auto" for automatic language detection with AWS Transcribe

## Output

The script creates:
1. An `audio` directory containing extracted audio
2. An `output` directory containing:
   - Video segments (.mp4 files)
   - Corresponding transcription text files (.txt)

## Notes

- AWS Transcribe requires an S3 bucket for processing. Make sure you have proper AWS permissions set up.
- Video segments are named with their timestamp ranges for easy reference.
- Both services provide timestamps for accurate video segmentation.

## AWS Transcribe Setup

1. Create an AWS account if you don't have one
2. Create an S3 bucket to store audio files for transcription
3. Create an IAM user with the following permissions:
   - AmazonTranscribeFullAccess
   - AmazonS3FullAccess (or more restricted bucket-specific permissions)
4. Get the AWS access key and secret for this IAM user
5. Add these credentials to your .env file

## Example Output Structure

```
output/
  ├── segment_000_0.00_10.50.mp4
  ├── segment_000_0.00_10.50.txt
  ├── segment_001_10.51_20.30.mp4
  ├── segment_001_10.51_20.30.txt
  ...
```

Each segment's text file contains the transcribed speech for that portion of the video.
