import React, { useEffect, useState } from 'react';
import { View, Text, Button, StyleSheet, Image } from 'react-native';
import * as FileSystem from 'expo-file-system';
import * as ImageManipulator from 'expo-image-manipulator';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';

const OnnxTest = () => {
  const [session, setSession] = useState<InferenceSession | null>(null);
  const [predictions, setPredictions] = useState<string[]>([]);
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('Initializing ONNX Runtime...');

        // Load the ONNX model asset
        const modelAsset = Asset.fromModule(require('./assets/mobilenetv2-7.onnx'));
        await modelAsset.downloadAsync();

        const modelUri = modelAsset.localUri || modelAsset.uri;
        console.log('Resolved Model URI:', modelUri);

        const fileInfo = await FileSystem.getInfoAsync(modelUri);
        if (!fileInfo.exists) {
          throw new Error(`Model file not found at ${modelUri}`);
        }

        console.log('Model Asset:', modelAsset);
        console.log('File Info:', fileInfo);

        // Load model file into ONNX Runtime
        const session = await InferenceSession.create(modelUri);
        setSession(session);
        console.log('Model loaded successfully');
        setIsModelLoaded(true);
      } catch (error) {
        console.error('Error during ONNX initialization:', error);
      }
    };

    loadModel();
  }, []);

  const preprocessImage = async (imageUri: string) => {
    // Resize image to 224x224 and convert to base64
    const resizedImage = await ImageManipulator.manipulateAsync(
      imageUri,
      [{ resize: { width: 224, height: 224 } }],
      { format: ImageManipulator.SaveFormat.PNG, base64: true }
    );

    const base64Data = resizedImage.base64;
    if (!base64Data) {
      throw new Error('Failed to convert image to base64');
    }

    // Convert base64 to Unit8Array
    const binaryString = atob(base64Data);
    const imageData = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      imageData[i] = binaryString.charCodeAt(i);
    }

    return imageData;
  };

  const normalizeImage = (imageData: Uint8Array) => {
    const floatData = new Float32Array(imageData.length);
    for (let i = 0; i < imageData.length; i++) {
      floatData[i] = imageData[i] / 255.0; 
    }
    return floatData;
  };

  const runInference = async () => {
    if (!session) {
      console.error('ONNX Session is not loaded.');
      return;
    }

    try {
      // Preprocess the input image
      const imageAsset = Asset.fromModule(require('./assets/car.jpeg'));
      await imageAsset.downloadAsync();
      const resizedImageData = await preprocessImage(imageAsset.localUri || imageAsset.uri);
      const normalizedData = normalizeImage(resizedImageData);

      // Verify tensor size
      if (normalizedData.length !== 150528) {
        throw new Error(`Tensor size mismatch: expected 150528, got ${normalizedData.length}`);
      }

      // Create ONNX tensor
      const inputTensor = new Tensor('float32', normalizedData, [1, 3, 224, 224]);

      // Run inference
      const feeds = { [session.inputNames[0]]: inputTensor };
      const output = await session.run(feeds);

      // Process the output tensor
      console.log('Inference Output:', output);
      const outputTensor = Object.values(output)[0]; 
      const predictionsArray = Array.from(outputTensor.data as Float32Array); 

      // Convert predictions to strings and set them
      const formattedPredictions = predictionsArray.map((val) => val.toFixed(2));
      setPredictions(formattedPredictions);
    } catch (error) {
      console.error('Error during inference:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>ONNX Object Detection</Text>
      <Image source={require('./assets/car.jpeg')} style={styles.image} />
      <Button
        title={isModelLoaded ? 'Run Detection' : 'Loading Model...'}
        onPress={runInference}
        disabled={!isModelLoaded}
      />
      <View style={styles.results}>
        {predictions.length > 0 ? (
          predictions.map((p, idx) => (
            <Text key={idx}>{p}</Text>
          ))
        ) : (
          <Text>No objects detected yet.</Text>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f9f9f9',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  image: {
    width: 300,
    height: 200,
    marginVertical: 10,
  },
  results: {
    marginTop: 20,
  },
});

export default OnnxTest;
