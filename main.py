import os
import re
import whisper

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s]

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def main():
    model = whisper.load_model("base")
    videos_path = "videos"
    results_path = "result"
    os.makedirs(results_path, exist_ok=True)
    video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv")
    files = [f for f in os.listdir(videos_path) if f.lower().endswith(video_extensions)]
    total_files = len(files)
    for index, file_name in enumerate(files, start=1):
        video_file = os.path.join(videos_path, file_name)
        percentage = (index / total_files) * 100
        print(f"Processing video {index}/{total_files} ({percentage:.2f}%) - {file_name}")
        result = model.transcribe(video_file)
        base_name = os.path.splitext(file_name)[0]
        txt_file = os.path.join(results_path, base_name + ".txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            for segment in result.get("segments", []):
                start = segment["start"]
                end = segment["end"]
                segment_text = segment["text"].strip()
                sentences = split_into_sentences(segment_text)
                if sentences:
                    duration_per_sentence = (end - start) / len(sentences)
                    for i, sentence in enumerate(sentences):
                        sentence_start = start + i * duration_per_sentence
                        sentence_end = sentence_start + duration_per_sentence
                        f.write(f"[{format_time(sentence_start)} - {format_time(sentence_end)}] {sentence}\n")
                else:
                    f.write(f"[{format_time(start)} - {format_time(end)}] {segment_text}\n")
        print(f"Transcription saved to: {txt_file}")

if __name__ == "__main__":
    main()
