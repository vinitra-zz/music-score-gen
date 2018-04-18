from __future__ import unicode_literals
import youtube_dl

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}

list_of_urls = ['https://youtube.com/watch?v=J1ly7q17-tY', 'https://www.youtube.com/watch?v=qKI0RpljZ5A']

for url in list_of_urls:
	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
		ydl.download([url])