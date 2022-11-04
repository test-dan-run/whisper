import requests

FILEPATH = 'test_commonvoice.wav'
LANGUAGE = 'Indonesian'

# translate from specified language to English
response = requests.post('http://0.0.0.0:8000/translate', 
    params={'filename': FILEPATH, 'language': LANGUAGE}, 
    files={'file': open(FILEPATH, 'rb')},
    )
print(response.json())

# transcribe to specified language
# note: models with ".en" are not equipped with multilingual transcription
response = requests.post('http://0.0.0.0:8000/transcribe', 
    params={'filename': FILEPATH, 'language': LANGUAGE}, 
    files={'file': open(FILEPATH, 'rb')},
    )

print(response.json())