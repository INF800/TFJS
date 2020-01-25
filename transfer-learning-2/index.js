// ----------------------------------------------------------------------------------------
// ---------------------------[ Helpers start ]--------------------------------------------
// ----------------------------------------------------------------------------------------

/**********************************[---UI Realted---]*************************************/

/**
 * Display / hide train dataset on click
 */
function showDiv() {
  document.getElementById('train-dataset').style.display = "block";
}
function hideDiv() {
  document.getElementById('train-dataset').style.display = "none";
}


/**
 * Display training process
 */
function myconsole() {
  if (truncatedMobileNet){
    let text = `
    Truncated model loaded \n`;
    document.getElementById('console').innerText = text;

    if (controllerDataset.xs != null){
      text = text + `Data Added \n`;
      document.getElementById('console').innerText = text;

      if (trainStarted != null){
        text = text + `Training started \n`;
        document.getElementById('console').innerText = text;
      }
    }
  }

  if (1==2) {
    // example
    document.getElementById('console').innerText = `
    prediction: ${result[0].className}\n
    probability: ${result[0].probability}
  `;
  }
}



/** 
* Function to make execution wait for given number
* of milliseconds.
* wait(ms)
*/
function wait(ms){
  var start = new Date().getTime();
  var end = start;
  while(end < start + ms) {
    end = new Date().getTime();
 }
}

/**********************************[---ML Realted---]*****************************************/
/**
 * USAGE: `img = capture(i.toString())` where `i` is INT
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
async function getImage(imgId) {
    //const img = await webcam.capture();
    picture = document.getElementById(imgId);
    // check if this line is required
    const img = tf.browser.fromPixels(picture);
    
    const processedImg =
        tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
    img.dispose();
    return processedImg;
  }

/**
 * A dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
class ControllerDataset {
    constructor(numClasses) {
      this.numClasses = numClasses;
    }
  
    /**
     * Adds an example to the controller dataset.
     * @param {Tensor} example A tensor representing the example. It can be an image,
     *     an activation, or any other type of Tensor.
     * @param {number} label The label of the example. Should be a number.
     */
    addExample(example, label) {
      // One-hot encode the label.
      const y = tf.tidy(
          () => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));
  
      if (this.xs == null) {
        // For the first example that gets added, keep example and y so that the
        // ControllerDataset owns the memory of the inputs. This makes sure that
        // if addExample() is called in a tf.tidy(), these Tensors will not get
        // disposed.
        this.xs = tf.keep(example);
        this.ys = tf.keep(y);
      } else {
        const oldX = this.xs;
        this.xs = tf.keep(oldX.concat(example, 0));
  
        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));
  
        oldX.dispose();
        oldY.dispose();
        y.dispose();
      }
    }
  }
  

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadTruncatedMobileNet() {
    const mobilenet = await tf.loadLayersModel(
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  
    // Return a model that outputs an internal activation.
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
  }

// When the UI buttons are pressed, read a frame from the webcam and associate
// it with the class label given by the button. up, down, left, right are
// labels 0, 1, 2, 3 respectively.
/*
ui.setExampleHandler(async label => {
    let img = await getImage("1");
  
    controllerDataset.addExample(truncatedMobileNet.predict(img), label);
  
    // Draw the preview thumbnail.
    ui.drawThumb(img, label);
    img.dispose();
  })
*/



/** 
 * Initialize dataset controller here as
 * `addExampleHandler` needs it
*/
const NUM_CLASSES = 3
// create instance for Dataset Controller to
// add data
controllerDataset = new ControllerDataset(NUM_CLASSES);
// adds images to dataset from html tag `ids` and `labels`
// note: Input size must be 224x224 in tags. 
async function addExampleHandler(label, imgId) {
    let img = await getImage(imgId);
    controllerDataset.addExample(truncatedMobileNet.predict(img), label);
    img.dispose();
}




/**
 * Sets up and trains the classifier.
 */
 async function train() {
    if (controllerDataset.xs == null) {
      throw new Error('Add some examples before training!');
    }
  
    // Creates a 2-layer fully connected model. By creating a separate model,
    // rather than adding layers to the mobilenet model, we "freeze" the weights
    // of the mobilenet model, and only train weights from the new model.
    const HiddenDenseUnits = 100
    model = tf.sequential({
      layers: [
        // Flattens the input to a vector so we can use it in a dense layer. While
        // technically a layer, this only performs a reshape (and has no training
        // parameters).
        tf.layers.flatten(
            {inputShape: truncatedMobileNet.outputs[0].shape.slice(1)}),
        // Layer 1.
        tf.layers.dense({
          units: HiddenDenseUnits,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
          useBias: true
        }),
        // Layer 2. The number of units of the last layer should correspond
        // to the number of classes we want to predict.
        tf.layers.dense({
          units: NUM_CLASSES,
          kernelInitializer: 'varianceScaling',
          useBias: false,
          activation: 'softmax'
        })
      ]
    });
    
    // hyperparams
    const LR = 1e-4
    const BatchSizeFraction = 0.4
    const Epochs = 20


    // Creates the optimizers which drives training of the model.
    const optimizer = tf.train.adam(LR);
    // We use categoricalCrossentropy which is the loss function we use for
    // categorical classification which measures the error between our predicted
    // probability distribution over classes (probability that an input is of each
    // class), versus the label (100% probability in the true class)>
    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  
    // We parameterize batch size as a fraction of the entire dataset because the
    // number of examples that are collected depends on how many examples the user
    // collects. This allows us to have a flexible batch size.
    const batchSize =
        Math.floor(controllerDataset.xs.shape[0] * BatchSizeFraction);
    if (!(batchSize > 0)) {
      throw new Error(
          `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
    }
    
    // We'll keep a buffer of loss and accuracy values over time.
    let trainBatchCount = 0;

    // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
    model.fit(controllerDataset.xs, controllerDataset.ys, {
      batchSize,
      epochs: Epochs,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          //console.log('loss:' + logs.loss.toFixed(5));
          trainBatchCount = trainBatchCount + 1
          plotLoss(trainBatchCount, logs.loss, 'train');

        }
      }
    });

    return 1;
  }
  
/**
 * To predict class based on image tag id
 * Note:
 * Input must be 1x224x224x3
 * `imgID` is of type INT
 */
async function predict(imgId){
  // load image using helper function
  const imgToPredict = await getImage(imgId);

  // Make a prediction through mobilenet, getting the internal activation of
  // the mobilenet model, i.e., "embeddings" of the input images.
  const embeddings = truncatedMobileNet.predict(imgToPredict);

  // Make a prediction through our newly-trained model using the embeddings
  // from mobilenet as input.
  const predictions = model.predict(embeddings);

  // Returns the index with the maximum probability. This number corresponds
  // to the class the model thinks is the most probable given the input.
  const predictedClass = predictions.as1D().argMax();
  const classId = (await predictedClass.data())[0];
  imgToPredict.dispose();

  return classId;
} 

/**********************************[---Visualisations---]***********************************/


const lossLabelElement = document.getElementById('loss-label');
//const accuracyLabelElement = document.getElementById('accuracy-label');
const lossValues = [[], [{x: 0, y: 0}]]; // dummy validation value

function plotLoss(batch, loss, set) {
  const series = set === 'train' ? 0 : 1;
  lossValues[series].push({x: batch, y: loss});
  lossValues[1].push({x: 0, y: 0}); // dummy
  console.log(lossValues)
  
  const lossContainer = document.getElementById('loss-canvas');
  tfvis.render.linechart(
      lossContainer, {values: lossValues, series: ['train', 'validation']}, {
        xLabel: 'Batch #',
        yLabel: 'Loss',
        width: 400,
        height: 300,
      });
  /*
  lossLabelElement.innerText = `last loss: ${loss.toFixed(3)}`;
  */
}

// ----------------------------------------------------------------------------------------
// ---------------------------[ Helpers end ]----------------------------------------------
// ----------------------------------------------------------------------------------------

/**
 * Main function
 */
async function app(){
    // load model
    trainStarted = false
    truncatedMobileNet = await loadTruncatedMobileNet();
    console.log("truncated model loaded.")
    myconsole();
    
    // add data into 
    console.log("adding data...")
    for (let i = 1; i<31; i++){
        // coke can -> 0
        if (i > 0 && i < 11 ){
          // label, id
          await addExampleHandler(0, i.toString())
        } 
        // pizza -> 1
        else if (i > 10 && i < 21 ) {
          await addExampleHandler(1, i.toString())
        } 
        // burger -> 2 
        else if (i > 20 && i < 31 ) {
          await addExampleHandler(1, i.toString())
        } 
        else {
          console.log("Loop incorrect")
        }
    }

    console.log("data added!")
    myconsole();
    // should be false!
    console.log(controllerDataset.xs == null)

    // train
    console.log("Training started");
    trainStarted = true
    await train()
    // `training completed!` will be displayed first and 
    // then training will start :/
    console.log('traing completed!')
    myconsole();
    
    
    // check what `warmup` is all about [X]

    // predict
    // -------
    // TODO: Make it wait!
    // wait(7000); // wait 7 secs
    // until completely trained. 
    // const predClass = await predict("9")
    // console.log(predClass)
    

    
}


//app();