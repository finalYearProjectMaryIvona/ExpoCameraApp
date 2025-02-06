import React, { useState, useEffect, useRef } from "react";
import { StyleSheet, Text, TouchableOpacity, View, Dimensions } from "react-native";
import { CameraView, CameraType, useCameraPermissions } from "expo-camera";
import { InferenceSession, Tensor } from "onnxruntime-react-native";
import * as ImageManipulator from "expo-image-manipulator";
import { Asset } from "expo-asset";
import Svg, { Rect, Text as SvgText } from 'react-native-svg';

const CLASSES: string[] = [ /* Class names remain the same */ ];
const COLORS = ['#FF3B30', '#34C759', '#007AFF', '#5856D6', '#FF9500'];

interface Detection {
  class: number;
  confidence: number;
  bbox: number[];
}

export default function App() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [permission, requestPermission] = useCameraPermissions();
  const [session, setSession] = useState<InferenceSession | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [modelLoading, setModelLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const cameraRef = useRef<any>(null);

  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    try {
      console.log("Loading YOLO model...");
      const modelAsset = Asset.fromModule(require("./assets/yolov8n.onnx")); // Use the raw model
      await modelAsset.downloadAsync();
      const session = await InferenceSession.create(modelAsset.localUri || modelAsset.uri);
      console.log("Model loaded successfully");
      setSession(session);
      setModelLoading(false);
    } catch (err) {
      console.error("Failed to load model:", err);
      setError("Failed to load model");
      setModelLoading(false);
    }
  };

  const preprocessImage = async (uri: string): Promise<Tensor> => {
    const resized = await ImageManipulator.manipulateAsync(
      uri,
      [{ resize: { width: 640, height: 640 } }],
      { base64: true }
    );

    if (!resized.base64) throw new Error("Failed to process image");

    const imageData = atob(resized.base64);
    const buffer = new Uint8Array(imageData.length);
    for (let i = 0; i < imageData.length; i++) {
      buffer[i] = imageData.charCodeAt(i);
    }

    const tensorData = new Float32Array(3 * 640 * 640);
    for (let i = 0; i < buffer.length; i += 3) {
      const r = buffer[i] / 255.0;
      const g = buffer[i + 1] / 255.0;
      const b = buffer[i + 2] / 255.0;

      const pixelIndex = Math.floor(i / 3);
      const row = Math.floor(pixelIndex / 640);
      const col = pixelIndex % 640;

      tensorData[row * 640 + col] = r;
      tensorData[640 * 640 + row * 640 + col] = g;
      tensorData[2 * 640 * 640 + row * 640 + col] = b;
    }

    return new Tensor("float32", tensorData, [1, 3, 640, 640]);
  };

  const runDetection = async () => {
    if (!session || !cameraRef.current || isProcessing) return;
    try {
      setIsProcessing(true);
      setError(null);
      const photo = await cameraRef.current.takePictureAsync({ quality: 1, base64: true, skipProcessing: true });
      const inputTensor = await preprocessImage(photo.uri);

      const results = await session.run({ images: inputTensor });
      const output = results.output0.data as Float32Array;
      let detections = processDetections(output, [...results.output0.dims]);
      setDetections(detections);
    } catch (err) {
      console.error("Detection failed:", err);
      setError("Detection failed");
    } finally {
      setIsProcessing(false);
    }
  };

  const processDetections = (output: Float32Array, dims: number[]) => {
    const [batch_size, num_values, num_boxes] = dims;
    let detections: Detection[] = [];
    for (let i = 0; i < num_boxes; i++) {
      const x1 = output[i * 4];
      const y1 = output[i * 4 + 1];
      const x2 = output[i * 4 + 2];
      const y2 = output[i * 4 + 3];
      const confidence = output[i * 5 + 4];
      const classId = output[i * 6 + 5];
      if (confidence > 0.5) {
        detections.push({ bbox: [x1, y1, x2, y2], confidence, class: classId });
      }
    }
    return detections;
  };

  const renderBoxes = () => {
    const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

    return (
      <Svg style={StyleSheet.absoluteFill}>
        {detections.map((det, idx) => {
          const [x1, y1, x2, y2] = det.bbox;
          const color = COLORS[idx % COLORS.length];
          const boxX = x1 * screenWidth;
          const boxY = y1 * screenHeight;
          const boxWidth = (x2 - x1) * screenWidth;
          const boxHeight = (y2 - y1) * screenHeight;

          return (
            <React.Fragment key={idx}>
              <Rect
                x={boxX}
                y={boxY}
                width={boxWidth}
                height={boxHeight}
                strokeWidth={2}
                stroke={color}
                fill="none"
              />
              <SvgText
                x={boxX}
                y={boxY - 5}
                fill={color}
                fontSize="14"
                fontWeight="bold"
              >
                {`${CLASSES[det.class]} ${(det.confidence * 100).toFixed(0)}%`}
              </SvgText>
            </React.Fragment>
          );
        })}
      </Svg>
    );
  };

  if (!permission?.granted) {
    return (
      <View style={styles.container}>
        <TouchableOpacity onPress={requestPermission}>
          <Text style={styles.text}>Grant Camera Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {modelLoading ? (
        <Text style={styles.text}>Loading model...</Text>
      ) : (
        <>
          <CameraView style={styles.camera} facing={facing} ref={cameraRef} />
          {renderBoxes()}
          <View style={styles.controls}>
            <TouchableOpacity style={styles.button} onPress={() => setFacing(f => f === 'back' ? 'front' : 'back')}>
              <Text style={styles.buttonText}>Flip</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.button, isProcessing && styles.buttonDisabled]} onPress={runDetection} disabled={isProcessing}>
              <Text style={styles.buttonText}>{isProcessing ? 'Processing...' : 'Detect'}</Text>
            </TouchableOpacity>
          </View>
          {error && <Text style={styles.error}>{error}</Text>}
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  camera: {
    flex: 1,
  },
  controls: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  button: {
    backgroundColor: '#4CAF50',
    padding: 15,
    borderRadius: 30,
    width: 100,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#888888',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  text: {
    color: 'white',
    fontSize: 18,
    textAlign: 'center',
    margin: 20,
  },
  error: {
    position: 'absolute',
    top: 20,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(255,0,0,0.7)',
    padding: 10,
    borderRadius: 5,
    color: 'white',
    textAlign: 'center',
  }
});
