import gradio as gr
import os
from main import VideoTranscriber
from pathlib import Path
import json
import logging
from datetime import datetime

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

def process_video(video_path, service, language, buffer, min_segment, method, num_frames, threshold, min_frame_diff):
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
                            if os.path.exists(frame_dir):
                                frames = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))]
                            
                            results.append({
                                "video": file_path,
                                "transcript": transcript,
                                "frames": frames
                            })

        yield "Processing completed."
        yield results

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise gr.Error(str(e))

def create_ui():
    with gr.Blocks(title="Video Transcriber") as app:
        gr.Markdown("""
        # Video Transcriber
        Upload a video to extract segments with speech, generate transcriptions, and extract key frames.
        """)
        
        with gr.Row():
            with gr.Column():
                video_input = gr.File(label="Upload Video")
                with gr.Row():
                    service = gr.Dropdown(choices=["aws"], value="aws", label="Transcription Service")
                    language = gr.Dropdown(choices=["Chinese", "English", "auto"], value="Chinese", label="Language")
                
                with gr.Row():
                    buffer = gr.Number(value=1.0, label="Buffer (seconds)")
                    min_segment = gr.Number(value=0.3, label="Minimum Segment Length (seconds)")
                
                with gr.Row():
                    method = gr.Dropdown(choices=["uniform", "random", "difference"], value="uniform", label="Frame Extraction Method")
                    num_frames = gr.Number(value=3, label="Number of Frames", precision=0)
                
                with gr.Row():
                    threshold = gr.Number(value=0.95, label="Frame Interval")
                    min_frame_diff = gr.Number(value=0.5, label="Frame Interval (seconds)")
                
                process_btn = gr.Button("Process Video", interactive=True)
                log_output = gr.Textbox(label="Processing Log", lines=10, max_lines=10)

            with gr.Column():
                status_output = gr.Markdown("Ready to process video")
                segments_list = gr.Dataframe(
                    headers=["Segment", "Duration", "Transcript"],
                    label="Processed Segments",
                    interactive=False
                )
                gallery = gr.Gallery(label="Extracted Frames")
                video_output = gr.Video(label="Video Segment")
                text_output = gr.Textbox(label="Transcription", lines=3)

        # Store results globally for access in callbacks
        results_store = gr.State([])

        def update_outputs(results):
            if not results:
                return None, None, None, None, None
            
            # Format results for dataframe
            segments_data = []
            for i, segment in enumerate(results):
                # Extract timestamp from transcript
                transcript = segment.get("transcript", "")
                timestamp = transcript.split(" ")[0] if transcript else "N/A"
                segments_data.append([
                    f"Segment {i+1}",
                    timestamp,
                    transcript
                ])
            
            # Display first segment's results by default
            first_segment = results[0]
            return (
                results,  # Store results
                segments_data,  # Dataframe
                first_segment["frames"],  # Gallery
                first_segment["video"],  # Video path
                first_segment["transcript"]  # Text
            )

        def on_segment_select(evt: gr.SelectData, stored_results):
            if not stored_results:
                return None, None, None
            
            # evt.index is a list containing [row, column], we want the row index
            row_idx = evt.index[0]
            if row_idx >= len(stored_results):
                return None, None, None
            
            segment = stored_results[row_idx]
            return (
                segment["frames"],  # Gallery
                segment["video"],  # Video path
                segment["transcript"]  # Text
            )

        def process_with_status(video, service_val, lang, buf, min_seg, 
                              method_val, frames, thresh, frame_diff):
            if video is None:
                raise gr.Error("Please upload a video file first")
            try:
                # Consume all progress messages
                progress_gen = process_video(
                    video, service_val, lang, buf, min_seg, method_val,
                    frames, thresh, frame_diff
                )
                results = None
                for item in progress_gen:
                    results = item  # The last item will be the results
                return results
            except Exception as e:
                raise gr.Error(str(e))

        def reset_gallery():
            return [],None

        def process_with_progress(video, service_val, lang, buf, min_seg, method_val, frames, thresh, frame_diff):
            # Clear all outputs
            log_output.value = ""  # Clear previous logs
            # gallery.update([])  # Clear gallery
            # video_output.update(value=None)  # Clear video
            # text_output.update(value="")  # Clear text
            # segments_list.update(value=[])  # Clear segments list
            try:
                progress_gen = process_video(video, service_val, lang, buf, min_seg, method_val, frames, thresh, frame_diff)
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
            outputs=[gallery,video_output]
        ).then(
            process_with_progress,  # Process video with progress updates
            inputs=[video_input, service, language, buffer, min_segment,
                   method, num_frames, threshold, min_frame_diff],
            outputs=[log_output, results_store],
            show_progress=True
        ).then(
            update_outputs,  # Update UI with results
            inputs=[results_store],
            outputs=[results_store, segments_list, gallery, video_output, text_output]
        ).then(
            lambda: ("Ready to process another video", gr.update(interactive=True)),  # Reset status and enable button
            None,
            [status_output, process_btn],
            queue=False
        )

        segments_list.select(
            on_segment_select,
            inputs=[results_store],
            outputs=[gallery, video_output, text_output]
        )

    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch()
