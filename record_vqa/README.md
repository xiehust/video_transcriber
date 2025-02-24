# Car Record and VQA

This tool extracts key-frame from input video and classify them into specific automotive components. Then, by building the knowledge bases, this tool supports the VQA about the video.

## Features
- Key-frame extraction
- Image classification
- Video QA


## Usage

- Step 1. Extract Key-frame

  ```bash
  python extract_video_frames_vqa.py ${PATH_TO_VIDEO} ${OUTPUT_DIR}
  ```

- Step 2. Image Classification

  ```bash
  python image_classify_vqa.py --folder ${PATH_TO_IMAGE_FOLDER} --s3_bucket ${S3_BUCKET} --threshold ${BLUR_THRESHOLD}
  ```

- Step 3. Build Knowledge Base
  - AWS Console: Build a Bedrock Knowledge Base with Data on `${S3_BUCKET}`
  - (TODO) AWS SDK

- Step 4. Video QA
  - Open WebUI by `python webui.py`, and switch to Video QA Table. Input the knowledge base id obtained in Step 3, and begin your QA!
