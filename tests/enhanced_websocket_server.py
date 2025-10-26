# enhanced_websocket_server.py
# Enhanced WebSocket server with latest API support and debugging tools
# Updated to handle WebSocket 13.0+ API changes

import asyncio
import websockets
import json
import numpy as np
import time
import logging
from typing import Set, Dict, Any


class EnhancedMeshStreamServer:
    """
    Enhanced WebSocket server that streams deformed mesh data to web client.
    Updated for WebSocket 13.0+ API (no path argument in handlers).
    """
    
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Performance tracking
        self.messages_sent = 0
        self.bytes_sent = 0
        self.start_time = time.time()
        
        # Debug mode
        self.debug_mode = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client."""
        self.clients.add(websocket)
        self.logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send welcome message
        welcome_msg = {
            'type': 'welcome',
            'message': 'Connected to BodyPix Cage Deformation Server',
            'timestamp': time.time(),
            'server_info': {
                'version': '1.0.0',
                'features': ['mesh_streaming', 'debug_mode', 'performance_stats']
            }
        }
        
        try:
            await websocket.send(json.dumps(welcome_msg))
        except websockets.ConnectionClosed:
            await self.unregister(websocket)
    
    async def unregister(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client."""
        self.clients.discard(websocket)
        self.logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_mesh_data(self, vertices: np.ndarray, faces: np.ndarray, 
                           metadata: Dict[str, Any] = None,
                           cage_vertices: np.ndarray = None,
                           cage_faces: np.ndarray = None):
        """
        Send mesh data and optional cage to all connected clients.
        
        Args:
            vertices: Nx3 array of mesh vertices
            faces: Mx3 array of mesh faces
            metadata: Optional metadata dict
            cage_vertices: Optional Nx3 array of cage vertices
            cage_faces: Optional Mx3 array of cage faces
        """
        if not self.clients:
            print("⚠️ No WebSocket clients connected, skipping mesh send")
            return None
        
        # Prepare data
        data = {
            'type': 'mesh_update',
            'vertices': vertices.tolist(),
            'faces': faces.tolist(),
            'timestamp': time.time(),
            'vertex_count': len(vertices),
            'face_count': len(faces)
        }
        
        # Add cage data if provided
        if cage_vertices is not None and cage_faces is not None:
            data['cage_vertices'] = cage_vertices.tolist()
            data['cage_faces'] = cage_faces.tolist()
            data['has_cage'] = True
        else:
            data['has_cage'] = False
        
        # Add metadata if provided
        if metadata:
            data['metadata'] = metadata
        
        # Add performance info if debug mode
        if self.debug_mode:
            data['performance'] = {
                'messages_sent': self.messages_sent,
                'bytes_sent': self.bytes_sent,
                'uptime': time.time() - self.start_time,
                'active_clients': len(self.clients)
            }
        
        # Debug: Check data types before serialization
        try:
            # Serialize message
            message = json.dumps(data)
            message_bytes = message.encode('utf-8')
        except TypeError as e:
            print(f"❌ JSON serialization error: {e}")
            print(f"Data types in metadata: {[(k, type(v)) for k, v in metadata.items()] if metadata else 'No metadata'}")
            # Try to fix numpy types
            if metadata:
                for key, value in metadata.items():
                    if hasattr(value, 'dtype'):  # numpy array
                        metadata[key] = value.tolist()
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if hasattr(subvalue, 'dtype'):
                                metadata[key][subkey] = subvalue.tolist()
            # Try again
            message = json.dumps(data)
            message_bytes = message.encode('utf-8')
        
        # Send to all clients
        if self.clients:
            await websockets.broadcast(self.clients, message)
            
            # Update stats
            self.messages_sent += 1
            self.bytes_sent += len(message_bytes)
    
    async def send_debug_info(self, debug_data: Dict[str, Any]):
        """
        Send debug information to clients.
        
        Args:
            debug_data: Dict containing debug information
        """
        if not self.clients:
            return
        
        data = {
            'type': 'debug_info',
            'timestamp': time.time(),
            'debug_data': debug_data
        }
        
        message = json.dumps(data)
        
        if self.clients:
            await websockets.broadcast(self.clients, message)
    
    async def send_performance_stats(self):
        """Send performance statistics to clients."""
        if not self.clients:
            return
        
        uptime = time.time() - self.start_time
        avg_messages_per_sec = self.messages_sent / uptime if uptime > 0 else 0
        avg_bytes_per_sec = self.bytes_sent / uptime if uptime > 0 else 0
        
        stats = {
            'type': 'performance_stats',
            'timestamp': time.time(),
            'stats': {
                'uptime': uptime,
                'messages_sent': self.messages_sent,
                'bytes_sent': self.bytes_sent,
                'active_clients': len(self.clients),
                'avg_messages_per_sec': avg_messages_per_sec,
                'avg_bytes_per_sec': avg_bytes_per_sec
            }
        }
        
        message = json.dumps(stats)
        
        if self.clients:
            await websockets.broadcast(self.clients, message)
    
    async def handle_client_message(self, websocket: websockets.WebSocketServerProtocol, 
                                  message: str):
        """
        Handle incoming message from client.
        
        Args:
            websocket: Client WebSocket connection
            message: Received message string
        """
        try:
            data = json.loads(message)
            msg_type = data.get('type', 'unknown')
            
            if msg_type == 'ping':
                # Respond to ping with pong
                pong_msg = {
                    'type': 'pong',
                    'timestamp': time.time()
                }
                await websocket.send(json.dumps(pong_msg))
            
            elif msg_type == 'request_debug':
                # Send debug information
                debug_data = {
                    'server_status': 'running',
                    'clients_connected': len(self.clients),
                    'debug_mode': self.debug_mode
                }
                await self.send_debug_info(debug_data)
            
            elif msg_type == 'request_performance':
                # Send performance statistics
                await self.send_performance_stats()
            
            elif msg_type == 'toggle_debug':
                # Toggle debug mode
                self.debug_mode = not self.debug_mode
                response = {
                    'type': 'debug_toggled',
                    'debug_mode': self.debug_mode,
                    'timestamp': time.time()
                }
                await websocket.send(json.dumps(response))
            
            else:
                self.logger.warning(f"Unknown message type: {msg_type}")
        
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON received from client")
        except Exception as e:
            self.logger.error(f"Error handling client message: {e}")
    
    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol):
        """
        Handle WebSocket connection.
        Updated for WebSocket 13.0+ API (no path argument).
        """
        await self.register(websocket)
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"Error in connection handler: {e}")
        finally:
            await self.unregister(websocket)
    
    async def start(self):
        """Start the WebSocket server."""
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        # Start server with updated API (no path argument)
        async with websockets.serve(self.handle_connection, self.host, self.port):
            self.logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            
            # Start performance monitoring task
            asyncio.create_task(self.performance_monitor())
            
            # Keep server running
            await asyncio.Future()  # Run forever
    
    async def performance_monitor(self):
        """Monitor and log performance statistics."""
        while True:
            await asyncio.sleep(30)  # Every 30 seconds
            
            uptime = time.time() - self.start_time
            avg_messages_per_sec = self.messages_sent / uptime if uptime > 0 else 0
            
            self.logger.info(
                f"Performance: {self.messages_sent} messages sent, "
                f"{self.bytes_sent} bytes sent, "
                f"{avg_messages_per_sec:.2f} msg/sec, "
                f"{len(self.clients)} clients"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current server statistics."""
        uptime = time.time() - self.start_time
        return {
            'uptime': uptime,
            'messages_sent': self.messages_sent,
            'bytes_sent': self.bytes_sent,
            'active_clients': len(self.clients),
            'avg_messages_per_sec': self.messages_sent / uptime if uptime > 0 else 0,
            'avg_bytes_per_sec': self.bytes_sent / uptime if uptime > 0 else 0
        }


# Standalone server function
def run_server(host='localhost', port=8765, debug=False):
    """
    Run the enhanced WebSocket server.
    
    Args:
        host: Server host address
        port: Server port
        debug: Enable debug mode
    """
    server = EnhancedMeshStreamServer(host, port)
    server.debug_mode = debug
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced WebSocket Server")
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    run_server(args.host, args.port, args.debug)
