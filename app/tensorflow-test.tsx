import React, { useState, useEffect } from 'react';
import { View, Text, Button, StyleSheet, Image } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Asset } from 'expo-asset';
import { decodeJpeg } from '@tensorflow/tfjs-react-native'; // Decoding JPEG files

export default function TensorFlowTest() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [imageAsset, setImageAsset] = useState<Asset | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        console.log('Initializing TensorFlow.js...');
        await tf.ready(); // Initialize TensorFlow.js
        console.log('TensorFlow.js initialized.');

        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
        console.log('COCO-SSD model loaded successfully.');

        // Preload image asset
        const asset = Asset.fromModule(require('./car.jpeg'));
        await asset.downloadAsync();
        setImageAsset(asset);
        console.log('Image preloaded:', asset.localUri);
      } catch (error) {
        console.error('Error initializing TensorFlow or loading model:', error);
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  const runDetection = async () => {
    if (!model || !imageAsset?.localUri) {
      console.error('Model not loaded or image asset missing.');
      return;
    }

    try {
      console.log('Processing image...');
      const response = await fetch(imageAsset.localUri);
      const rawImageData = await response.arrayBuffer();
      const imageTensor = decodeJpeg(new Uint8Array(rawImageData)); // Decode image

      console.log('Running object detection...');
      const predictions = await model.detect(imageTensor);
      console.log('Predictions:', predictions);
      setPredictions(predictions);

      imageTensor.dispose();
    } catch (error) {
      console.error('Error during detection:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>TensorFlow Object Detection</Text>
      {imageAsset && (
        <Image
          source={{ uri: imageAsset.localUri }}
          style={styles.image}
          resizeMode="contain"
        />
      )}
      <Button
        title={isLoading ? "Loading Model..." : "Run Detection"}
        onPress={runDetection}
        disabled={isLoading || !model}
      />
      <View style={styles.results}>
        {predictions.length > 0 ? (
          predictions.map((p, idx) => (
            <Text key={idx}>{`${p.class}: ${Math.round(p.score * 100)}%`}</Text>
          ))
        ) : (
          <Text>No objects detected yet.</Text>
        )}
      </View>
    </View>
  );
}

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
