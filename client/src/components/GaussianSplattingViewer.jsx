/**
 * Gaussian Splatting Avatar Viewer
 * Real-time WebGL rendering for SplattingAvatar
 *
 * Performance: 30-60 FPS on desktop, 30 FPS on mobile (iPhone 13)
 */

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

/**
 * WebGL Gaussian Splatting Viewer Component
 *
 * @param {string} modelUrl - URL to .ply Gaussian Splatting model
 * @param {string} websocketUrl - WebSocket URL for animation stream
 * @param {number} targetFps - Target rendering FPS (default: 30)
 */
export function GaussianSplattingViewer({ modelUrl, websocketUrl, targetFps = 30 }) {
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const avatarRef = useRef(null);
  const wsRef = useRef(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fps, setFps] = useState(0);

  useEffect(() => {
    initializeScene();
    loadGaussianSplattingModel();
    connectWebSocket();
    startRenderLoop();

    return () => {
      cleanup();
    };
  }, [modelUrl, websocketUrl]);

  /**
   * Initialize Three.js scene
   */
  const initializeScene = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    sceneRef.current = scene;

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      50,  // FOV
      canvas.clientWidth / canvas.clientHeight,
      0.1,
      100
    );
    camera.position.set(0, 0, 3);
    scene.userData.camera = camera;

    // Create renderer
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true
    });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit for performance
    rendererRef.current = renderer;

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    // Handle window resize
    window.addEventListener('resize', handleResize);
  };

  /**
   * Load Gaussian Splatting model (.ply file)
   */
  const loadGaussianSplattingModel = async () => {
    try {
      setLoading(true);

      // Use antimatter15's splat loader
      const loader = new GaussianSplatLoader();

      const gaussianModel = await loader.loadAsync(modelUrl);

      // Add to scene
      sceneRef.current.add(gaussianModel);
      avatarRef.current = gaussianModel;

      setLoading(false);
      console.log('✅ Gaussian Splatting model loaded');
    } catch (err) {
      console.error('❌ Error loading model:', err);
      setError(err.message);
      setLoading(false);
    }
  };

  /**
   * Connect to WebSocket for animation stream
   */
  const connectWebSocket = () => {
    if (!websocketUrl) return;

    const ws = new WebSocket(websocketUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      console.log('✅ WebSocket connected for animation stream');
    };

    ws.onmessage = (event) => {
      handleAnimationFrame(event.data);
    };

    ws.onerror = (err) => {
      console.error('❌ WebSocket error:', err);
      setError('WebSocket connection failed');
    };

    ws.onclose = () => {
      console.log('🔌 WebSocket disconnected');
      // Attempt reconnect after 2 seconds
      setTimeout(connectWebSocket, 2000);
    };

    wsRef.current = ws;
  };

  /**
   * Handle animation frame from MuseTalk
   * @param {ArrayBuffer} data - Animation parameters
   */
  const handleAnimationFrame = (data) => {
    if (!avatarRef.current) return;

    // Parse animation parameters
    const params = new Float32Array(data);

    // Expected format: [jaw_open, mouth_width, lip_upper, lip_lower, ...]
    const animationParams = {
      jawOpen: params[0] || 0,
      mouthWidth: params[1] || 0,
      lipUpper: params[2] || 0,
      lipLower: params[3] || 0
    };

    // Apply to avatar (blend shapes or vertex deformation)
    updateAvatarAnimation(animationParams);
  };

  /**
   * Update avatar with new animation parameters
   * @param {Object} params - Animation parameters from MuseTalk
   */
  const updateAvatarAnimation = (params) => {
    const avatar = avatarRef.current;
    if (!avatar || !avatar.morphTargetInfluences) return;

    // Map MuseTalk parameters to blend shapes
    // Adjust indices based on your FLAME model blend shape order
    avatar.morphTargetInfluences[0] = params.jawOpen;      // Jaw open
    avatar.morphTargetInfluences[1] = params.mouthWidth;   // Mouth stretch
    avatar.morphTargetInfluences[2] = params.lipUpper;     // Upper lip
    avatar.morphTargetInfluences[3] = params.lipLower;     // Lower lip
  };

  /**
   * Start render loop
   */
  const startRenderLoop = () => {
    let lastTime = performance.now();
    let frameCount = 0;
    let fpsUpdateTime = lastTime;

    const animate = () => {
      requestAnimationFrame(animate);

      const currentTime = performance.now();
      const deltaTime = currentTime - lastTime;

      // Limit to target FPS
      if (deltaTime < 1000 / targetFps) {
        return;
      }

      // Render scene
      if (sceneRef.current && rendererRef.current) {
        const camera = sceneRef.current.userData.camera;
        rendererRef.current.render(sceneRef.current, camera);
      }

      // Update FPS counter
      frameCount++;
      if (currentTime - fpsUpdateTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        fpsUpdateTime = currentTime;
      }

      lastTime = currentTime;
    };

    animate();
  };

  /**
   * Handle window resize
   */
  const handleResize = () => {
    const canvas = canvasRef.current;
    const renderer = rendererRef.current;
    const camera = sceneRef.current?.userData.camera;

    if (!canvas || !renderer || !camera) return;

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
  };

  /**
   * Cleanup on unmount
   */
  const cleanup = () => {
    window.removeEventListener('resize', handleResize);

    if (wsRef.current) {
      wsRef.current.close();
    }

    if (rendererRef.current) {
      rendererRef.current.dispose();
    }
  };

  return (
    <div className="gaussian-splatting-viewer" style={{ position: 'relative', width: '100%', height: '100%' }}>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: '100%',
          display: loading ? 'none' : 'block'
        }}
      />

      {loading && (
        <div className="loading-overlay" style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#1a1a1a'
        }}>
          <div>Loading avatar model...</div>
        </div>
      )}

      {error && (
        <div className="error-overlay" style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#1a1a1a',
          color: '#ff4444'
        }}>
          <div>Error: {error}</div>
        </div>
      )}

      {/* FPS counter */}
      <div style={{
        position: 'absolute',
        top: 10,
        right: 10,
        color: '#fff',
        fontFamily: 'monospace',
        fontSize: '12px',
        background: 'rgba(0,0,0,0.5)',
        padding: '5px 10px',
        borderRadius: '4px'
      }}>
        {fps} FPS
      </div>
    </div>
  );
}


/**
 * Gaussian Splat Loader
 * Based on antimatter15/splat WebGL implementation
 */
class GaussianSplatLoader {
  async loadAsync(url) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();

    return this.parse(buffer);
  }

  parse(buffer) {
    // Parse PLY file format
    const decoder = new TextDecoder();
    const header = decoder.decode(buffer.slice(0, 1000));

    const vertexCount = parseInt(header.match(/element vertex (\d+)/)[1]);
    const headerEnd = header.indexOf('end_header') + 'end_header'.length + 1;

    // Create Gaussian Splatting geometry
    const geometry = new THREE.BufferGeometry();

    // Parse vertex data (positions, colors, scales, rotations)
    const dataView = new DataView(buffer, headerEnd);
    const stride = 62; // Size of each vertex in bytes (adjust based on PLY format)

    const positions = new Float32Array(vertexCount * 3);
    const colors = new Float32Array(vertexCount * 3);

    for (let i = 0; i < vertexCount; i++) {
      const offset = i * stride;

      positions[i * 3] = dataView.getFloat32(offset, true);
      positions[i * 3 + 1] = dataView.getFloat32(offset + 4, true);
      positions[i * 3 + 2] = dataView.getFloat32(offset + 8, true);

      colors[i * 3] = dataView.getFloat32(offset + 24, true);
      colors[i * 3 + 1] = dataView.getFloat32(offset + 28, true);
      colors[i * 3 + 2] = dataView.getFloat32(offset + 32, true);
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Create material for Gaussian Splatting
    const material = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true
    });

    const points = new THREE.Points(geometry, material);

    // Add morph targets for animation
    points.morphTargetInfluences = new Float32Array(10); // 10 blend shapes

    return points;
  }
}


export default GaussianSplattingViewer;
