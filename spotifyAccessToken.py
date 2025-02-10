import requests

client_id = 'your-client-id'
client_secret = 'your-client-secret'

url = 'https://accounts.spotify.com/api/token'
headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
}

data = {
    'grant_type': 'client_credentials',
    'client_id': client_id,
    'client_secret': client_secret
}

response = requests.post(url, headers=headers, data=data)

if response.status_code == 200:
    access_token = response.json().get('access_token')
    print(f'Access Token: {access_token}')
    print(' ---------- ')
    print(f'Response: {response.json()}')
else:
    print(f'Failed to get access token. Status code: {response.status_code}')
    print(f'Response: {response.text}')
