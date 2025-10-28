"""
Enhanced WebSocket Server V2
=============================

Improved WebSocket server with:
1. Better error handling
2. Compatible with websockets 13.0+
3. Cage data support
4. Connection state management

Author: AI Assistant
Date: October 26, 2025
"""

import asyncio
import websockets
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMeshStreamServerV2:
    """
    WebSocket server for streaming mesh and cage data to web clients.
    """
    
    def __init__(self, host='localhost', port=8765):
        """
        Initialize server.
        
        Args:
            host: Host address
            port: Port number
        """
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
    
    async def handle_client(self, websocket):
        """
        Handle a single client connection.
        
        Args:
            websocket: WebSocket connection
        """
        # Register client
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Client connected from {client_addr}. Total clients: {len(self.clients)}")
        
        try:
            # Keep connection alive
            async for message in websocket:
                # Handle any incoming messages (e.g., control commands)
                try:
                    data = json.loads(message)
                    logger.info(f"Received from client: {data}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_addr} disconnected normally")
        
        except Exception as e:
            logger.error(f"Error handling client {client_addr}: {e}")
        
        finally:
            # Unregister client
            self.clients.discard(websocket)
            logger.info(f"Client {client_addr} removed. Total clients: {len(self.clients)}")
    
    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
    
    async def send_mesh_data(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        cage_vertices: np.ndarray = None,
        cage_faces: np.ndarray = None,
        debug_data: dict = None
    ):
        """
        Send mesh and cage data to all connected clients.
        
        Args:
            vertices: (N, 3) mesh vertex positions
            faces: (M, 3) mesh face indices
            cage_vertices: Optional (K, 3) cage vertex positions
            cage_faces: Optional (L, 3) cage face indices
        """
        if len(self.clients) == 0:
            return
        
        # Prepare data
        data = {
            'type': 'mesh_update',
            'vertices': vertices.tolist(),
            'faces': faces.tolist(),
            'timestamp': asyncio.get_event_loop().time(),
            'vertex_count': len(vertices),
            'face_count': len(faces),
            'metadata': {
                'has_cage': cage_vertices is not None
            }
        }
        
        # Add cage data if provided
        if cage_vertices is not None and cage_faces is not None:
            data['cage_vertices'] = cage_vertices.tolist()
            data['cage_faces'] = cage_faces.tolist()
            data['cage_vertex_count'] = len(cage_vertices)
            data['cage_face_count'] = len(cage_faces)
        
        # Add debug data if provided
        if debug_data is not None:
            data['debug'] = debug_data
        
        # Convert to JSON
        try:
            message = json.dumps(data)
        except Exception as e:
            logger.error(f"Error serializing mesh data: {e}")
            return
        
        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
    
    async def broadcast(self, message_type: str, data: dict):
        """
        Broadcast a generic message to all clients.
        
        Args:
            message_type: Type of message
            data: Message data
        """
        if len(self.clients) == 0:
            return
        
        message = json.dumps({
            'type': message_type,
            **data
        })
        
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                disconnected.add(client)
        
        self.clients -= disconnected
    
    def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            logger.info("WebSocket server stopped")


# ============================================================================
# Testing
# ============================================================================

async def test_server():
    """Test the WebSocket server."""
    server = EnhancedMeshStreamServerV2()
    await server.start()
    
    print("Server running. Press Ctrl+C to stop.")
    
    # Keep running
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == "__main__":
    print("="*60)
    print("Enhanced WebSocket Server V2 - Test")
    print("="*60)
    print("\nStarting test server on ws://localhost:8765")
    print("Connect a web client to test.")
    print("="*60 + "\n")
    
    asyncio.run(test_server())

