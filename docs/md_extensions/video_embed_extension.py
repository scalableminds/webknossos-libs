from markdown import Extension
from markdown.inlinepatterns import Pattern

VIDEO_LINK_RE = r'\!\[(?P<alt>[^\]]*)\]\((https?://(www\.|)'\
                r'(youtube\.com/watch\?\S*v=(?P<video_id>[A-Za-z0-9_=-]+)(&t=(?P<start>[A-Za-z0-9_&=-]+)s)?)\S*)' \
                r'(?<!png)(?<!jpg)(?<!jpeg)(?<!gif)\)'\

class VideoEmbedExtension(Extension):
  """
  Embed Vimeo and Youtube videos in python markdown by using ![alt text](vimeo or youtube url)
  """
  def extendMarkdown(self, md):
    link_pattern = VideoLink(VIDEO_LINK_RE, md)
    link_pattern.ext = self
    md.inlinePatterns.add('video_embed', link_pattern, '<image_link')

class VideoLink(Pattern):
  def handleMatch(self, match):
    alt = match.group("alt").strip()
    video_id = match.group("video_id")
    start = match.group("start")
    if video_id:
      html = self.make_iframe(video_id.strip(), alt, start)
      return self.markdown.htmlStash.store(html)
    return None

  def make_iframe(self, video_id, alt, start):
    url = f"https://www.youtube-nocookie.com/embed/{video_id}"
    if start:
      start_time = int(start.strip())
      url += f"?start={start_time}"
    return (
      f"<iframe class='youtube' src='{url}' alt='{alt}' "
      +"frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture' "
      + "style='width: 500px; width: -webkit-fill-available; width: fill-available; width: stretch; aspect-ratio: 1.91;' allowfullscreen></iframe>"
    )

def makeExtension(**kwargs):
    return VideoEmbedExtension(**kwargs)
