import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Button, Alert } from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Asset } from 'expo-asset';
import * as FileSystem from 'expo-file-system';

export default function HomeScreen() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [output, setOutput] = useState('');

  // Paths for TensorFlow Lite assets
  const modelUri = FileSystem.documentDirectory + 'model.tflite';
  const labelsUri = FileSystem.documentDirectory + 'labels.txt';
  const imageUri = FileSystem.documentDirectory + 'car.jpeg';

  useEffect(() => {
    (async () => {
      try {
        console.log('Starting TensorFlow Lite setup...');
        const modelAsset = Asset.fromModule(require('./model.tflite'));
        const labelsAsset = Asset.fromModule(require('./labels.txt'));
        const imageAsset = Asset.fromModule(require('./car.jpeg'));

        await modelAsset.downloadAsync();
        await labelsAsset.downloadAsync();
        await imageAsset.downloadAsync();

        console.log('Assets downloaded:', {
          model: modelAsset.localUri,
          labels: labelsAsset.localUri,
          image: imageAsset.localUri,
        });

        await FileSystem.copyAsync({
          from: modelAsset.localUri!,
          to: modelUri,
        });

        await FileSystem.copyAsync({
          from: labelsAsset.localUri!,
          to: labelsUri,
        });

        await FileSystem.copyAsync({
          from: imageAsset.localUri!,
          to: imageUri,
        });

        console.log('Files copied successfully to FileSystem:', {
          modelUri,
          labelsUri,
          imageUri,
        });

        // Mock TensorFlow Lite setup
        console.log('Initializing TensorFlow Lite...');
        console.log('TensorFlow Lite initialized successfully!');
      } catch (error) {
        console.error('Error during TensorFlow Lite setup:', error);
      }
    })();
  }, []);

  const analyzeImage = async () => {
    try {
      console.log('Analyzing image from:', imageUri);

      // Mock TensorFlow Lite functionality
      const mockRunModelOnImage = (config: any, callback: (err: any, res: any) => void) => {
        console.log('Mock runModelOnImage called with config:', config);
        const mockResults = [
          { label: 'Car', confidence: 0.85 },
          { label: 'Bike', confidence: 0.76 },
        ];
        callback(null, mockResults);
      };

      mockRunModelOnImage(
        {
          path: imageUri,
          imageMean: 0,
          imageStd: 255,
          numResults: 5,
          threshold: 0.5,
        },
        (err: any, res: any) => {
          if (err) {
            console.error('Error running TensorFlow Lite model:', err);
            Alert.alert('Error', 'Error analyzing image.');
          } else {
            console.log('Model output:', res);
            setOutput(JSON.stringify(res, null, 2));
          }
        }
      );
    } catch (error) {
      console.error('Error analyzing image:', error);
    }
  };

  if (!permission) {
    return <View style={styles.container}><Text>Loading permissions...</Text></View>;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="Grant Permission" />
      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  }

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing}>
        <View style={styles.buttonContainer}>
          <Button title="Flip Camera" onPress={toggleCameraFacing} />
          <Button title="Analyze Image" onPress={analyzeImage} />
        </View>
      </CameraView>
      <Text style={styles.output}>{output}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 64,
    justifyContent: 'space-between',
  },
  output: {
    textAlign: 'center',
    margin: 10,
    fontSize: 14,
    color: 'black',
  },
});
