from transformers import pipeline
import gradio as gr

pipe = pipeline(model="ayoubkirouane/whisper-small-ar")  

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="Whisper Small Arabic",
    description="Realtime demo for Arabic speech recognition using a fine-tuned Whisper small model.",
)
iface.launch()