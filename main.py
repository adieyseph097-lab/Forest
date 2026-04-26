"""
Adieyseph AI - Multi-feature AI Application for Termux/Android
Features: Chatbot, Image Recognition, Text Generation, Voice Processing
"""

import os
import json
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.core.window import Window
from kivy.clock import Clock
import threading

# AI Libraries
try:
    from transformers import pipeline
    import torch
except ImportError:
    print("Installing required packages...")
    os.system("pip install transformers torch")

try:
    import speech_recognition as sr
except ImportError:
    os.system("pip install SpeechRecognition")

try:
    from PIL import Image as PILImage
except ImportError:
    os.system("pip install Pillow")

# Set window size for mobile
Window.size = (360, 640)

class AIFeatures:
    """AI Features Handler"""
    
    def __init__(self):
        self.chatbot = None
        self.text_generator = None
        self.image_classifier = None
        self.speech_recognizer = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all AI models"""
        try:
            print("Loading Chatbot model...")
            self.chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
            print("Chatbot loaded!")
        except Exception as e:
            print(f"Chatbot error: {e}")
        
        try:
            print("Loading Text Generation model...")
            self.text_generator = pipeline("text-generation", model="distilgpt2")
            print("Text Generator loaded!")
        except Exception as e:
            print(f"Text Generator error: {e}")
        
        try:
            print("Loading Image Classification model...")
            self.image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
            print("Image Classifier loaded!")
        except Exception as e:
            print(f"Image Classifier error: {e}")
        
        try:
            self.speech_recognizer = sr.Recognizer()
            print("Speech Recognizer loaded!")
        except Exception as e:
            print(f"Speech Recognizer error: {e}")
    
    def chat(self, user_input):
        """Generate chatbot response"""
        if self.chatbot:
            try:
                response = self.chatbot(user_input)
                return response[0]['generated_text']
            except Exception as e:
                return f"Error: {str(e)}"
        return "Chatbot not available"
    
    def generate_text(self, prompt, max_length=100):
        """Generate text from prompt"""
        if self.text_generator:
            try:
                result = self.text_generator(prompt, max_length=max_length, num_return_sequences=1)
                return result[0]['generated_text']
            except Exception as e:
                return f"Error: {str(e)}"
        return "Text Generator not available"
    
    def classify_image(self, image_path):
        """Classify image"""
        if self.image_classifier:
            try:
                results = self.image_classifier(image_path)
                return results
            except Exception as e:
                return f"Error: {str(e)}"
        return "Image Classifier not available"
    
    def recognize_speech(self):
        """Recognize speech from microphone"""
        if self.speech_recognizer:
            try:
                with sr.Microphone() as source:
                    audio = self.speech_recognizer.listen(source, timeout=5)
                    text = self.speech_recognizer.recognize_google(audio)
                    return text
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError:
                return "API error"
            except Exception as e:
                return f"Error: {str(e)}"
        return "Speech Recognizer not available"


class AdieysephAIApp(App):
    """Main AI Application"""
    
    def build(self):
        self.title = "Adieyseph AI"
        self.ai = AIFeatures()
        
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        title = Label(text="🤖 Adieyseph AI", size_hint_y=0.1, bold=True, font_size='20sp')
        main_layout.add_widget(title)
        
        # Feature selector
        self.feature_spinner = Spinner(
            text='Select Feature',
            values=('Chatbot', 'Text Generation', 'Image Recognition', 'Voice Processing'),
            size_hint_y=0.1
        )
        main_layout.add_widget(self.feature_spinner)
        
        # Input area
        self.input_field = TextInput(
            hint_text='Enter your input here...',
            multiline=True,
            size_hint_y=0.2
        )
        main_layout.add_widget(self.input_field)
        
        # Button layout
        button_layout = GridLayout(cols=2, size_hint_y=0.15, spacing=10)
        
        process_btn = Button(text='Process', background_color=(0.2, 0.6, 0.8, 1))
        process_btn.bind(on_press=self.process_input)
        button_layout.add_widget(process_btn)
        
        voice_btn = Button(text='🎤 Voice Input', background_color=(0.8, 0.2, 0.2, 1))
        voice_btn.bind(on_press=self.voice_input)
        button_layout.add_widget(voice_btn)
        
        clear_btn = Button(text='Clear', background_color=(0.6, 0.6, 0.6, 1))
        clear_btn.bind(on_press=self.clear_input)
        button_layout.add_widget(clear_btn)
        
        settings_btn = Button(text='⚙️ Settings', background_color=(0.4, 0.4, 0.4, 1))
        settings_btn.bind(on_press=self.open_settings)
        button_layout.add_widget(settings_btn)
        
        main_layout.add_widget(button_layout)
        
        # Output area
        scroll = ScrollView(size_hint_y=0.55)
        self.output_field = Label(
            text='Output will appear here...',
            size_hint_y=None,
            markup=True,
            text_size=(320, None)
        )
        self.output_field.bind(texture_size=self.output_field.setter('size'))
        scroll.add_widget(self.output_field)
        main_layout.add_widget(scroll)
        
        return main_layout
    
    def process_input(self, instance):
        """Process user input based on selected feature"""
        feature = self.feature_spinner.text
        user_input = self.input_field.text.strip()
        
        if not user_input and feature != 'Image Recognition':
            self.output_field.text = "[color=ff0000]Please enter input![/color]"
            return
        
        self.output_field.text = "[color=ffaa00]Processing...[/color]"
        
        # Run in thread to prevent UI freeze
        thread = threading.Thread(
            target=self._process_feature,
            args=(feature, user_input)
        )
        thread.daemon = True
        thread.start()
    
    def _process_feature(self, feature, user_input):
        """Process AI feature in background thread"""
        try:
            if feature == 'Chatbot':
                result = self.ai.chat(user_input)
            elif feature == 'Text Generation':
                result = self.ai.generate_text(user_input)
            elif feature == 'Image Recognition':
                result = str(self.ai.classify_image(user_input))
            elif feature == 'Voice Processing':
                result = self.ai.recognize_speech()
            else:
                result = "Feature not selected"
            
            # Update UI from main thread
            Clock.schedule_once(
                lambda dt: setattr(
                    self.output_field,
                    'text',
                    f"[color=00ff00]✓ Result:[/color]\n{result}"
                ),
                0
            )
        except Exception as e:
            Clock.schedule_once(
                lambda dt: setattr(
                    self.output_field,
                    'text',
                    f"[color=ff0000]Error: {str(e)}[/color]"
                ),
                0
            )
    
    def voice_input(self, instance):
        """Handle voice input"""
        self.output_field.text = "[color=ffaa00]Listening...[/color]"
        
        thread = threading.Thread(target=self._voice_process)
        thread.daemon = True
        thread.start()
    
    def _voice_process(self):
        """Process voice in background"""
        result = self.ai.recognize_speech()
        Clock.schedule_once(
            lambda dt: (
                setattr(self.input_field, 'text', result),
                setattr(self.output_field, 'text', f"[color=00ff00]Recognized:[/color]\n{result}")
            ),
            0
        )
    
    def clear_input(self, instance):
        """Clear input and output"""
        self.input_field.text = ""
        self.output_field.text = "Output will appear here..."
    
    def open_settings(self, instance):
        """Open settings popup"""
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(text='Settings', bold=True, size_hint_y=0.2))
        content.add_widget(Label(text='Model: Lightweight AI Models\nVersion: 1.0.0', size_hint_y=0.6))
        
        close_btn = Button(text='Close', size_hint_y=0.2)
        content.add_widget(close_btn)
        
        popup = Popup(title='Settings', content=content, size_hint=(0.9, 0.6))
        close_btn.bind(on_press=popup.dismiss)
        popup.open()


if __name__ == '__main__':
    app = AdieysephAIApp()
    app.run()