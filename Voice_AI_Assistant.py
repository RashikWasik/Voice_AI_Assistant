import gradio as gr
import speech_recognition as sr
from gtts import gTTS
from groq import Groq
import tempfile

# CONFIG 
GROQ_API_KEY = "Put Your Own API Key"

client = Groq(api_key=GROQ_API_KEY)
recognizer = sr.Recognizer()

# CORE FUNCTION 
def process_voice(audio_path, history):
    status = ""
    
    if audio_path is None:
        return history, None, "Please record audio first!", gr.update(value=None)
    
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en-US")
            
        if not text or not text.strip():
            return history, None, "No speech detected (empty voice). Please speak clearly.", gr.update(value=None)
            
    except sr.UnknownValueError:
        return history, None, "Couldn't understand (gibberish/quiet). Try again!", gr.update(value=None)
    except sr.RequestError:
        return history, None, "Speech service error. Check internet.", gr.update(value=None)
    except Exception as e:
        return history, None, f"STT error: {str(e)}", gr.update(value=None)

    messages = [
        {"role": "system", "content": "You are a friendly voice AI assistant. Answer naturally and keep replies short."}
    ]
    
    history = history or []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": text})

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        ai_response = completion.choices[0].message.content.strip()
    except Exception as e:
        ai_response = f"Sorry, AI error: {str(e)}"

    new_history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": ai_response}
    ]

    try:
        tts = gTTS(text=ai_response, lang="en", slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        tmp.close()
        voice_path = tmp.name
    except Exception:
        voice_path = None
        ai_response += " (voice playback failed)"

    return new_history, voice_path, "", gr.update(value=None)


# GRADIO UI 
with gr.Blocks(title="Voice AI Assistant (Made by RW)") as demo:
    gr.Markdown("""
    # Voice AI Assistant
    ### <span style="color: #ea580c;">(Made by Rashik Wasik)</span>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=300,
                avatar_images=["https://api.dicebear.com/7.x/lorelei/svg?seed=sherlock", "https://api.dicebear.com/7.x/bottts/png?seed=Robot14"]   # humanic person + robot
            )
        
        with gr.Column(scale=1):
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )

    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="🎙️ Speak here",
        visible=True,
        buttons=[]          
    )

    with gr.Row():
        with gr.Column(scale=1):
            send_button = gr.Button("✅ Send Voice to AI", variant="primary", size="large")
        with gr.Column(scale=1):
            clear_button = gr.Button("🗑️ Clear Conversation", variant="secondary", size="large")

    response_audio = gr.Audio(
        label="🔊 AI Voice Response",
        autoplay=True,
        buttons=[]
    )

    # SEND BUTTON LOGIC 
    send_button.click(
        fn=process_voice,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, response_audio, status_box, audio_input]
    )

    clear_button.click(
        fn=lambda: ([], None, "", gr.update(value=None)),
        outputs=[chatbot, response_audio, status_box, audio_input]
    )

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    show_error=True
)
