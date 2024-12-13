declare module 'react-native-tflite' {
    export default class TFLite {
      loadModel(
        params: { model: string; labels: string },
        callback: (error: any) => void
      ): void;
  
      runModelOnImage(
        params: {
          path: string;
          imageMean: number;
          imageStd: number;
          numResults: number;
          threshold: number;
        },
        callback: (error: any, response: any) => void
      ): void;
    }
  }
  