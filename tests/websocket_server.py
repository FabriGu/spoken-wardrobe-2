# websocket_server.py
# WebSocket server to stream mesh data to Three.js frontend

import asyncio
import websockets
import json
import numpy as np


class MeshStreamServer:
    """
    WebSocket server that streams deformed mesh data to web client.
    """
    
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        
    async def register(self, websocket):
        """Register a new client."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
    
    async def unregister(self, websocket):
        """Unregister a client."""
        self.clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_mesh_data(self, vertices, faces, keypoints=None):
        if not self.clients:
            return
        data = {
            'type': 'mesh_update',
            'vertices': vertices.tolist(),
            'faces': faces.tolist()
        }
        if keypoints is not None:
            data['keypoints'] = keypoints
        message = json.dumps(data)
        websockets.broadcast(self.clients, message)
    
    async def handler(self, websocket):
        await self.register(websocket)
        try:
            while True:
                # Wait for a message or just sleep to keep alive
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=60)
                    # Optionally handle msg
                except asyncio.TimeoutError:
                    # No message, just keep connection alive
                    continue
        except websockets.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
    
    async def start(self):
        """Start the WebSocket server."""
        # Use a wrapper to ensure correct handler signature
        async def handler(websocket):
            await self.handler(websocket)
        async with websockets.serve(handler, self.host, self.port):
            print(f"WebSocket server started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


# Standalone server function
def run_server():
    """Run the WebSocket server."""
    server = MeshStreamServer()
    asyncio.run(server.start())


if __name__ == "__main__":
    run_server()
