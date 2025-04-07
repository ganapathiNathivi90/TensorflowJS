let model; // Store the model globally

// Load model only once
async function loadModel() {
    if (!model) {
        console.log("Loading model...");
        model = await tf.loadGraphModel(
            'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1',
            { fromTFHub: true }
        );
        console.log("Model loaded successfully!");
    }
}

// Preprocess image
async function preprocessImage(imgElement) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imgElement)  // Convert image to tensor
            .resizeNearestNeighbor([224, 224])  // Resize to 224x224
            .toFloat()  // Convert pixel values to float
            .div(127.5)  // Normalize (0-255 -> 0-2)
            .sub(1)      // Normalize (0-2 -> -1 to 1)
            .expandDims();  // Add batch dimension: [1, 224, 224, 3]
        
        return tensor;
    });
}

// Classify Image
async function classifyImage() {
    await loadModel();  // Ensure model is loaded
    const imgElement = document.getElementById('image');
    
    // Preprocess image
    const tensor = await preprocessImage(imgElement);
    
    // Run inference
    const prediction = await model.executeAsync(tensor);

    // Get the highest probability class index
    const classIndex = prediction.argMax(-1).dataSync()[0];

    console.log(prediction);
    
    document.getElementById("result").innerText = `Prediction: Class ${classIndex} name `;
    console.log("Predicted Class Index:", classIndex);
}

// Run classification when the page loads
window.onload = classifyImage;
