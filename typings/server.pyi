"""
Stub file for ComfyUI server module.
This file provides type hints for the server module that's only available when ComfyUI is running.
"""

from typing import Any, Dict, List, Optional, Union
from aiohttp import web

class PromptServer:
    """ComfyUI PromptServer class stub."""
    
    instance: 'PromptServer'
    
    def __init__(self) -> None:
        pass
    
    @property
    def routes(self) -> web.RouteTableDef:
        """Routes for the server."""
        pass
    
    def send_sync(self, event: str, data: Any, sid: Optional[str] = None) -> None:
        """Send synchronous event."""
        pass
    
    def send_async(self, event: str, data: Any, sid: Optional[str] = None) -> None:
        """Send asynchronous event."""
        pass

# Global instance
PromptServer.instance = PromptServer() 