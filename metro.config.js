const { getDefaultConfig } = require('expo/metro-config');

module.exports = (() => {
  const config = getDefaultConfig(__dirname);
  config.resolver.assetExts.push('jpeg', 'jpg', 'png', 'tflite', 'jpg', 'json', 'txt'); 
  return config;
})();
