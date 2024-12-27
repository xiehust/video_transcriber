import gradio as gr
import os
from main import VideoTranscriber
from pathlib import Path
import json
import logging
from datetime import datetime
from image_classify import ImageClassifier, PRO_MODEL_ID, LITE_MODEL_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogHandler(logging.Handler):
    def __init__(self, log_output):
        super().__init__()
        self.log_output = log_output
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_output.value = self.log_output.value + msg + "\n"
        except Exception:
            self.handleError(record)

def process_video(video_path, service, language, buffer, min_segment, method, num_frames, threshold, min_frame_diff, model_id):
    # Initialize image classifier
    image_classifier = ImageClassifier(model_id=model_id)
    try:
        # Create output directory based on video filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        video_name = os.path.basename(video_path.name).split('.')[0]
        output_dir = os.path.join("output", video_name,timestamp)
        frames_dir = os.path.join(output_dir, "frames")
        
        # Initialize video transcriber
        transcriber = VideoTranscriber(video_path.name, service=service, language=language)
        
        # Process video and yield progress messages
        for message in transcriber.process_video(buffer=buffer, 
                                               min_segment=min_segment,
                                               output_dir=output_dir):
            yield message
        
        # Extract frames
        yield "Extracting frames from video segments..."
        
        from extract_video_frames import extract_video_frames
        extract_video_frames(output_dir,
                           output_base_dir=frames_dir,
                           method=method,
                           num_frames=num_frames,
                           threshold=threshold,
                           min_frame_diff=min_frame_diff)

        yield "Collecting results..."
        # Collect results
        results = []
        seen_timestamps = set()  # Track unique timestamp ranges
        
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.mp4'):
                    # Find corresponding txt and image files
                    base_name = file[:-4]
                    txt_path = os.path.join(root, base_name + '.txt')
                    transcript = ""
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            transcript = f.read()
                            
                        # Extract timestamp from transcript
                        timestamp = transcript.split(" ")[0] if transcript else None
                        if timestamp and timestamp not in seen_timestamps:
                            seen_timestamps.add(timestamp)
                            
                            # Find related frames
                            frame_dir = os.path.join(frames_dir, base_name)
                            frames = []
                            frame_classifications = []
                            if os.path.exists(frame_dir):
                                frame_files = [f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))]
                                for frame_file in frame_files:
                                    frame_path = os.path.join(frame_dir, frame_file)
                                    frames.append(frame_path)
                                    # Classify each frame
                                    try:
                                        classification_result = image_classifier.process(frame_path)
                                        if classification_result:
                                            # Remove the ```json prefix and ``` suffix if present
                                            if classification_result.startswith('```json'):
                                                classification_result = classification_result[7:]
                                            if classification_result.endswith('```'):
                                                classification_result = classification_result[:-3]
                                            frame_classifications.append(json.loads(classification_result))
                                        else:
                                            frame_classifications.append({"error": classification_result})
                                    except Exception as e:
                                        logger.error(f"Error classifying frame {frame_path}: {e}")
                                        frame_classifications.append({"error": str(e)})
                            
                            results.append({
                                "video": file_path,
                                "transcript": transcript,
                                "frames": frames,
                                "classifications": frame_classifications
                            })

        yield "Processing completed."
        yield results

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise gr.Error(str(e))

def create_ui():
    with gr.Blocks(title="Video Transcriber and Analyzer") as app:
        gr.Markdown("""
        # Video Transcriber and Analyzer
        上传视频，截取初视频中有说话的片段，转录字幕，并抽取关键帧，识别图片内容
        """)
        
        with gr.Row():
            with gr.Column():
                video_input = gr.File(label="Upload Video")
                with gr.Row():
                    service = gr.Dropdown(choices=["aws"], value="aws", label="Transcription Service", visible=False)
                    language = gr.Dropdown(choices=["Chinese", "English", "auto"], value="Chinese", label="语言")
                
                with gr.Row():
                    buffer = gr.Number(value=1, label="视频片段前后添加窗口 (seconds)")
                    min_segment = gr.Number(value=0.5, label="视频片段最小长度(seconds)")
                
                with gr.Row():
                    method = gr.Dropdown(choices=["uniform", "random", "difference"], value="uniform", label="关键帧抽取方法")
                    num_frames = gr.Number(value=3, label="关键帧数量", precision=0)
                
                with gr.Row():
                    threshold = gr.Number(value=0.85, label="关键帧相似度阈值(只对difference方法)")
                    min_frame_diff = gr.Number(value=0.5, label="帧之间最小间隔(seconds)")
                
                with gr.Row():
                    model_id = gr.Dropdown(
                        choices=[
                            ("Nova Lite", LITE_MODEL_ID),
                            ("Nova Pro", PRO_MODEL_ID)
                        ],
                        value=LITE_MODEL_ID,
                        label="图像分类模型"
                    )
                
                process_btn = gr.Button("Process Video", interactive=True)
                log_output = gr.Textbox(label="Processing Log", lines=10, max_lines=10)

            with gr.Column():
                status_output = gr.Markdown("Ready to process video")
                segments_list = gr.Dataframe(
                    headers=["Segment", "Duration", "Transcript", "Classifications"],
                    label="Processed Segments",
                    interactive=False
                )
                gallery = gr.Gallery(label="Extracted Frames")
                video_output = gr.Video(label="Video Segment")
                text_output = gr.Textbox(label="Transcription", lines=3)
                classification_output = gr.Textbox(label="Classification Results", lines=5)

        # Store results globally for access in callbacks
        results_store = gr.State([])

        def update_outputs(results):
            if not results:
                return None, None, None, None, None, None
            
            # Format results for dataframe
            segments_data = []
            for i, segment in enumerate(results):
                # Extract timestamp from transcript
                transcript = segment.get("transcript", "")
                timestamp = transcript.split(" ")[0] if transcript else "N/A"
                # Format classifications for display
                classifications_text = ""
                if "classifications" in segment:
                    for j, classification in enumerate(segment["classifications"]):
                        if "error" in classification:
                            classifications_text += f"Frame {j+1}: Error - {classification['error']}\n"
                        else:
                            classifications_text += f"Frame {j+1}: {classification.get('category', 'N/A')} - {classification.get('sub_category', 'N/A')} (Confidence: {classification.get('confidence', 0)}%)\n"
                
                segments_data.append([
                    f"Segment {i+1}",
                    timestamp,
                    transcript,
                    classifications_text
                ])
            
            # Display first segment's results by default
            first_segment = results[0]
            # Show all frames' classifications
            first_classification = ""
            if "classifications" in first_segment:
                for i, classification in enumerate(first_segment["classifications"]):
                    first_classification += f"Frame {i+1}:\n"
                    if "error" in classification:
                        first_classification += f"Error: {classification['error']}\n"
                    else:
                        first_classification += f"Category: {classification.get('category', 'N/A')}\n"
                        first_classification += f"Sub-category: {classification.get('sub_category', 'N/A')}\n"
                        first_classification += f"Confidence: {classification.get('confidence', 0)}%\n"
                        if classification.get('comments'):
                            first_classification += f"Comments: {classification['comments']}\n"
                    first_classification += "\n"

            return (
                results,  # Store results
                segments_data,  # Dataframe
                first_segment["frames"],  # Gallery
                first_segment["video"],  # Video path
                first_segment["transcript"],  # Text
                first_classification  # Classification results
            )

        def on_segment_select(evt: gr.SelectData, stored_results):
            if not stored_results:
                return None, None, None, None
            
            # evt.index is a list containing [row, column], we want the row index
            row_idx = evt.index[0]
            if row_idx >= len(stored_results):
                return None, None, None, None
            
            segment = stored_results[row_idx]
            # Show all frames' classifications for selected segment
            classification_text = ""
            if "classifications" in segment:
                for i, classification in enumerate(segment["classifications"]):
                    classification_text += f"Frame {i+1}:\n"
                    if "error" in classification:
                        classification_text += f"Error: {classification['error']}\n"
                    else:
                        classification_text += f"Category: {classification.get('category', 'N/A')}\n"
                        classification_text += f"Sub-category: {classification.get('sub_category', 'N/A')}\n"
                        classification_text += f"Confidence: {classification.get('confidence', 0)}%\n"
                        if classification.get('comments'):
                            classification_text += f"Comments: {classification['comments']}\n"
                    classification_text += "\n"
            
            return (
                segment["frames"],  # Gallery
                segment["video"],  # Video path
                segment["transcript"],  # Text
                classification_text  # Classification results
            )

        def on_gallery_select(evt: gr.SelectData, stored_results):
            if not stored_results or not evt.index:
                return None
            
            # Find the current segment and frame
            for segment in stored_results:
                if len(segment["frames"]) > evt.index and len(segment["classifications"]) > evt.index:
                    classification = segment["classifications"][evt.index]
                    if "error" in classification:
                        return f"Error: {classification['error']}"
                    
                    result = f"Category: {classification.get('category', 'N/A')}\n"
                    result += f"Sub-category: {classification.get('sub_category', 'N/A')}\n"
                    result += f"Confidence: {classification.get('confidence', 0)}%\n"
                    if classification.get('comments'):
                        result += f"Comments: {classification['comments']}"
                    return result
            return None

        def reset_gallery():
            return [], None

        def process_with_progress(video, service_val, lang, buf, min_seg, method_val, frames, thresh, frame_diff, model_id):
            # Clear all outputs
            log_output.value = ""  # Clear previous logs
            try:
                progress_gen = process_video(video, service_val, lang, buf, min_seg, method_val, frames, thresh, frame_diff, model_id)
                results = None
                
                # Process all items from the generator
                for item in progress_gen:
                    if isinstance(item, str):
                        # This is a progress message
                        log_output.value += item + "\n"
                        yield [log_output.value, None]
                    else:
                        # This is the final results
                        results = item
                
                if results:
                    yield [log_output.value, results]
                else:
                    raise gr.Error("No results were generated")
                    
            except Exception as e:
                log_output.value += f"Error: {str(e)}\n"
                yield [log_output.value, None]
                raise gr.Error(str(e))

        process_btn.click(
            lambda: ("Processing video...", gr.update(interactive=False)),  # Set initial status, disable button
            None,
            [status_output, process_btn],
            queue=False
        ).then(
            reset_gallery,
            outputs=[gallery, video_output]
        ).then(
            process_with_progress,  # Process video with progress updates
            inputs=[video_input, service, language, buffer, min_segment,
                   method, num_frames, threshold, min_frame_diff, model_id],
            outputs=[log_output, results_store],
            show_progress=True
        ).then(
            update_outputs,  # Update UI with results
            inputs=[results_store],
            outputs=[results_store, segments_list, gallery, video_output, text_output, classification_output]
        ).then(
            lambda: ("Ready to process another video", gr.update(interactive=True)),  # Reset status and enable button
            None,
            [status_output, process_btn],
            queue=False
        )

        segments_list.select(
            on_segment_select,
            inputs=[results_store],
            outputs=[gallery, video_output, text_output, classification_output]
        )

        gallery.select(
            on_gallery_select,
            inputs=[results_store],
            outputs=[classification_output]
        )

    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(share=True)
