import requests

response = requests.post('http://0.0.0.0:8000/transcribe', params={'filename': 'test.wav', 'language': 'Indonesian'}, files={'file': open('test.wav', 'rb')})
print(response.json())