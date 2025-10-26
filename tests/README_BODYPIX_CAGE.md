# BodyPix Cage Deformation System

A real-time 3D mesh deformation system that uses BodyPix segmentation to create intelligent cage-based deformation for clothing overlay.

## Features

- **BodyPix Integration**: Uses BodyPix for precise body part segmentation
- **Cage-Based Deformation**: Intelligent cage generation based on segmented body parts
- **Real-Time Performance**: Optimized for live video processing
- **Web Rendering**: Three.js-based web viewer with debugging tools
- **WebSocket Streaming**: Real-time mesh data streaming to web browser

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install tf-bodypix
```

### 2. Run the Integrated System

```bash
python tests/test_integration.py
```

### 3. Open Web Viewer

Open `tests/enhanced_mesh_viewer.html` in your web browser.

## System Components

### Core Scripts

- `test_integration.py` - Main integration script
- `test_bodypix_cage_deformation.py` - Standalone BodyPix cage deformation
- `enhanced_cage_utils.py` - Enhanced cage generation utilities
- `enhanced_websocket_server.py` - WebSocket server with debugging
- `enhanced_mesh_viewer.html` - Web viewer with debugging tools

### Key Features

#### BodyPix Integration

- Real-time body part segmentation
- Support for 24 body parts
- Intelligent cage placement based on segmentation

#### Cage-Based Deformation

- Mean Value Coordinates (MVC) for smooth deformation
- Anatomical cage generation
- Real-time mesh deformation

#### Web Rendering

- Three.js-based 3D viewer
- Real-time mesh updates
- Debugging tools and performance monitoring
- WebSocket communication

## Controls

### Python Application

- `Q` - Quit
- `D` - Toggle debug mode
- `S` - Toggle segmentation display
- `R` - Reset cage
- `O` - Show web viewer instructions

### Web Viewer

- Debug panel toggle
- Wireframe mode
- Keypoints display
- Camera reset
- Performance statistics

## Architecture

```
Camera Input → BodyPix Segmentation → Cage Generation → Mesh Deformation → WebSocket → Web Viewer
```

1. **Camera Input**: Live video feed from webcam
2. **BodyPix Segmentation**: Extract body part masks
3. **Cage Generation**: Create anatomical cage based on segmentation
4. **Mesh Deformation**: Apply cage-based deformation using MVC
5. **WebSocket Streaming**: Send deformed mesh to web client
6. **Web Rendering**: Display mesh in Three.js viewer

## Debugging Tools

### Python Side

- Performance timing for each component
- Debug mode with detailed logging
- Segmentation visualization
- Cage vertex display

### Web Side

- Real-time performance charts
- Connection status monitoring
- Mesh statistics
- Server performance metrics

## Performance

- **BodyPix Processing**: ~50-100ms per frame
- **Cage Generation**: ~10-20ms per frame
- **Mesh Deformation**: ~5-10ms per frame
- **Total Pipeline**: ~65-130ms per frame
- **Target FPS**: 10-15 FPS (suitable for real-time)

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**

   - Ensure port 8765 is available
   - Check firewall settings
   - Verify server is running

2. **BodyPix Model Loading**

   - First run downloads model (~50MB)
   - Ensure internet connection
   - Check tf-bodypix installation

3. **Camera Not Working**
   - Check camera permissions
   - Try different camera index
   - Verify OpenCV installation

### Debug Mode

Enable debug mode for detailed logging:

```bash
python tests/test_integration.py --debug
```

## Future Enhancements

- [ ] Depth estimation integration
- [ ] Multiple clothing types support
- [ ] Improved cage generation algorithms
- [ ] Real-time performance optimization
- [ ] Mobile device support

## Dependencies

- Python 3.8+
- OpenCV
- NumPy
- Trimesh
- tf-bodypix
- websockets
- scipy
- Three.js (web)

## License

This project is part of the spoken_wardrobe_2 system.
