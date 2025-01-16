import React, { useState, useEffect } from 'react';
import { View, Text, Button, StyleSheet, Image } from 'react-native';
import Tflite from 'react-native-tflite';
import { Asset } from 'expo-asset';

const TFLiteTest = () => {
  const [tflite, setTflite] = useState<any>(null);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('Initializing TFLite...');
        const tfliteInstance = new Tflite();
        setTflite(tfliteInstance);

        // Load TFLite model and labels from the assets folder
        const modelAsset = Asset.fromModule(require('./assets/model.tflite'));
        const labelsAsset = Asset.fromModule(require('./assets/labels.txt'));

        // Ensure the assets are available locally
        await modelAsset.downloadAsync();
        await labelsAsset.downloadAsync();

        const modelPath = modelAsset.localUri || modelAsset.uri;
        const labelPath = labelsAsset.localUri || labelsAsset.uri;

        console.log('Model path:', modelPath);
        console.log('Labels path:', labelPath);

        tfliteInstance.loadModel(
          {
            model: modelPath,
            labels: labelPath,
          },
          (error: any) => {
            if (error) {
              console.error('Error loading model:', error);
            } else {
              console.log('Model loaded successfully');
              setIsModelLoaded(true);
            }
          }
        );
      } catch (error) {
        console.error('Error during TFLite initialization:', error);
      }
    };

    loadModel();

    return () => {
      if (tflite) {
        tflite.close();
      }
    };
  }, []);

  const runDetection = async () => {
    if (!isModelLoaded || !tflite) {
      console.error('Model is not loaded yet.');
      return;
    }

    console.log('Running detection...');
    tflite.runModelOnImage(
      {
        path: require('./assets/car.jpeg'), // Image to analyze
        imageMean: 0.0,
        imageStd: 255.0,
        numResults: 5,
        threshold: 0.5,
      },
      (error: any, result: any) => {
        if (error) {
          console.error('Error during detection:', error);
        } else {
          console.log('Detection results:', result);
          setPredictions(result || []);
        }
      }
    );
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>TFLite Object Detection</Text>
      <Image source={require('./assets/car.jpeg')} style={styles.image} />
      <Button
        title={isModelLoaded ? 'Run Detection' : 'Loading Model...'}
        onPress={runDetection}
        disabled={!isModelLoaded}
      />
      <View style={styles.results}>
        {predictions.length > 0 ? (
          predictions.map((p, idx) => (
            <Text key={idx}>{`${p.label}: ${Math.round(p.confidence * 100)}%`}</Text>
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

export default TFLiteTest;
