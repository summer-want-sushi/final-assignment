# tools/youtube_tools.py (Updated with fixes)
"""
YouTube Tools Module - Fixed version using pytubefix
Addresses network issues, deprecation warnings, and playlist errors
"""

from pytubefix import YouTube, Playlist
from pytubefix.cli import on_progress
from typing import Optional, Dict, Any, List
import os
import time
import logging
from .utils import logger, validate_file_exists

class YouTubeTools:
    """YouTube tools with improved error handling and network resilience"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.supported_formats = ['mp4', '3gp', 'webm']
        self.supported_audio_formats = ['mp3', 'mp4', 'webm']
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _retry_operation(self, operation, *args, **kwargs):
        """Retry operation with exponential backoff for network issues"""
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                
                error_msg = str(e).lower()
                if any(term in error_msg for term in ['network', 'socket', 'timeout', 'connection']):
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Network error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve comprehensive metadata about a YouTube video using pytubefix
        """
        try:
            def _get_info():
                yt = YouTube(url, on_progress_callback=on_progress)
                
                # Get available streams info with better error handling
                video_streams = []
                try:
                    streams = yt.streams.filter(progressive=True, file_extension='mp4')
                    for stream in streams:
                        try:
                            video_streams.append({
                                'resolution': getattr(stream, 'resolution', 'unknown'),
                                'fps': getattr(stream, 'fps', 'unknown'),
                                'video_codec': getattr(stream, 'video_codec', 'unknown'),
                                'audio_codec': getattr(stream, 'audio_codec', 'unknown'),
                                'filesize': getattr(stream, 'filesize', None),
                                'mime_type': getattr(stream, 'mime_type', 'unknown')
                            })
                        except Exception as stream_error:
                            logger.debug(f"Error processing stream: {stream_error}")
                            continue
                except Exception as e:
                    logger.warning(f"Could not retrieve stream details: {e}")
                
                # Get caption languages safely
                captions_available = []
                try:
                    if yt.captions:
                        captions_available = list(yt.captions.keys())
                except Exception as e:
                    logger.warning(f"Could not retrieve captions list: {e}")
                
                info = {
                    'title': getattr(yt, 'title', 'Unknown'),
                    'author': getattr(yt, 'author', 'Unknown'),
                    'channel_url': getattr(yt, 'channel_url', 'Unknown'),
                    'length': getattr(yt, 'length', 0),
                    'views': getattr(yt, 'views', 0),
                    'description': getattr(yt, 'description', ''),
                    'thumbnail_url': getattr(yt, 'thumbnail_url', ''),
                    'publish_date': yt.publish_date.isoformat() if getattr(yt, 'publish_date', None) else None,
                    'keywords': getattr(yt, 'keywords', []),
                    'video_id': getattr(yt, 'video_id', ''),
                    'watch_url': getattr(yt, 'watch_url', url),
                    'available_streams': video_streams,
                    'captions_available': captions_available
                }
                
                return info
            
            info = self._retry_operation(_get_info)
            if info is not None:
                logger.info(f"Retrieved info for video: {info.get('title', 'Unknown')}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info for {url}: {e}")
            return None
    
    def download_video(self, url: str, output_path: str = './downloads', 
                      resolution: str = 'highest', filename: Optional[str] = None) -> Optional[str]:
        """Download a YouTube video with retry logic"""
        try:
            def _download():
                os.makedirs(output_path, exist_ok=True)
                
                yt = YouTube(url, on_progress_callback=on_progress)
                
                # Select stream based on resolution preference
                if resolution == 'highest':
                    stream = yt.streams.get_highest_resolution()
                elif resolution == 'lowest':
                    stream = yt.streams.get_lowest_resolution()
                else:
                    stream = yt.streams.filter(res=resolution, progressive=True, file_extension='mp4').first()
                    if not stream:
                        logger.warning(f"Resolution {resolution} not found, downloading highest instead")
                        stream = yt.streams.get_highest_resolution()
                
                if not stream:
                    raise Exception("No suitable stream found for download")
                
                # Download with custom filename if provided
                if filename:
                    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    file_path = stream.download(output_path=output_path, filename=f"{safe_filename}.{stream.subtype}")
                else:
                    file_path = stream.download(output_path=output_path)
                
                return file_path
            
            file_path = self._retry_operation(_download)
            logger.info(f"Downloaded video to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download video from {url}: {e}")
            return None
    
    def download_audio(self, url: str, output_path: str = './downloads', 
                      filename: Optional[str] = None) -> Optional[str]:
        """Download only audio from a YouTube video with retry logic"""
        try:
            def _download_audio():
                os.makedirs(output_path, exist_ok=True)
                
                yt = YouTube(url, on_progress_callback=on_progress)
                audio_stream = yt.streams.get_audio_only()
                
                if not audio_stream:
                    raise Exception("No audio stream found")
                
                if filename:
                    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    file_path = audio_stream.download(output_path=output_path, filename=f"{safe_filename}.{audio_stream.subtype}")
                else:
                    file_path = audio_stream.download(output_path=output_path)
                
                return file_path
            
            file_path = self._retry_operation(_download_audio)
            logger.info(f"Downloaded audio to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download audio from {url}: {e}")
            return None
    
    def get_captions(self, url: str, language_code: str = 'en') -> Optional[str]:
        """
        Get captions/subtitles - FIXED: No more deprecation warning
        """
        try:
            def _get_captions():
                yt = YouTube(url, on_progress_callback=on_progress)
                
                if not yt.captions:
                    logger.warning("No captions available for this video")
                    return None
                
                # Use modern dictionary-style access instead of deprecated method
                if language_code in yt.captions:
                    caption = yt.captions[language_code]
                    captions_text = caption.generate_srt_captions()
                    return captions_text
                else:
                    available_langs = list(yt.captions.keys())
                    logger.warning(f"Captions not found for language {language_code}. Available: {available_langs}")
                    return None
            
            result = self._retry_operation(_get_captions)
            if result:
                logger.info(f"Retrieved captions in {language_code}")
            return result
                
        except Exception as e:
            logger.error(f"Failed to get captions from {url}: {e}")
            return None
    
    def get_playlist_info(self, playlist_url: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a YouTube playlist - FIXED: Better error handling
        """
        try:
            def _get_playlist_info():
                playlist = Playlist(playlist_url)
                
                # Get video URLs first (this triggers the playlist loading)
                video_urls = list(playlist.video_urls)
                
                # Safely access playlist properties with fallbacks
                info = {
                    'video_count': len(video_urls),
                    'video_urls': video_urls[:10],  # Limit to first 10 for performance
                    'total_videos': len(video_urls)
                }
                
                # Try to get additional info, but don't fail if unavailable
                try:
                    info['title'] = getattr(playlist, 'title', 'Unknown Playlist')
                except:
                    info['title'] = 'Private/Unavailable Playlist'
                
                try:
                    info['description'] = getattr(playlist, 'description', '')
                except:
                    info['description'] = 'Description unavailable'
                
                try:
                    info['owner'] = getattr(playlist, 'owner', 'Unknown')
                except:
                    info['owner'] = 'Owner unavailable'
                
                return info
            
            info = self._retry_operation(_get_playlist_info)
            if info is not None:
                logger.info(f"Retrieved playlist info: {info['title']} ({info['video_count']} videos)")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get playlist info from {playlist_url}: {e}")
            return None
    
    def get_available_qualities(self, url: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get all available download qualities - FIXED: Better network handling
        """
        try:
            def _get_qualities():
                yt = YouTube(url, on_progress_callback=on_progress)
                streams = []
                
                # Get progressive streams (video + audio)
                for stream in yt.streams.filter(progressive=True):
                    try:
                        streams.append({
                            'resolution': getattr(stream, 'resolution', 'unknown'),
                            'fps': getattr(stream, 'fps', 'unknown'),
                            'filesize_mb': round(stream.filesize / (1024 * 1024), 2) if getattr(stream, 'filesize', None) else None,
                            'mime_type': getattr(stream, 'mime_type', 'unknown'),
                            'video_codec': getattr(stream, 'video_codec', 'unknown'),
                            'audio_codec': getattr(stream, 'audio_codec', 'unknown')
                        })
                    except Exception as stream_error:
                        logger.debug(f"Error processing stream: {stream_error}")
                        continue
                
                # Sort by resolution (numeric part)
                def sort_key(x):
                    res = x['resolution']
                    if res and res != 'unknown' and res[:-1].isdigit():
                        return int(res[:-1])
                    return 0
                
                return sorted(streams, key=sort_key, reverse=True)
            
            return self._retry_operation(_get_qualities)
            
        except Exception as e:
            logger.error(f"Failed to get qualities for {url}: {e}")
            return None

# Convenience functions (unchanged)
def get_video_info(url: str) -> Optional[Dict[str, Any]]:
    """Standalone function to get video information"""
    tools = YouTubeTools()
    return tools.get_video_info(url)

def download_video(url: str, output_path: str = './downloads', 
                  resolution: str = 'highest', filename: Optional[str] = None) -> Optional[str]:
    """Standalone function to download a video"""
    tools = YouTubeTools()
    return tools.download_video(url, output_path, resolution, filename)

def download_audio(url: str, output_path: str = './downloads', 
                  filename: Optional[str] = None) -> Optional[str]:
    """Standalone function to download audio only"""
    tools = YouTubeTools()
    return tools.download_audio(url, output_path, filename)

def get_captions(url: str, language_code: str = 'en') -> Optional[str]:
    """Standalone function to get video captions"""
    tools = YouTubeTools()
    return tools.get_captions(url, language_code)

def get_playlist_info(playlist_url: str) -> Optional[Dict[str, Any]]:
    """Standalone function to get playlist information"""
    tools = YouTubeTools()
    return tools.get_playlist_info(playlist_url)
