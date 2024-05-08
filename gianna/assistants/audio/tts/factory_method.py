from gianna.assistants.audio.tts.factories import TextToSpeechFactory, TextToSpeechType


def text_to_speech(text: str, speech_type: str, lang='en', voice='default'):
    speech_type = TextToSpeechType(speech_type)
    factory = TextToSpeechFactory()
    tts = factory.create_text_to_speech(speech_type, language=lang, voice=voice)
    tts.synthesize(text)
