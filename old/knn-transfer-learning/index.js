async function start(){
  console.log("loading train data")
  // Load Train Data
  const imageTensors = [];
  const imgLabels = []
  // --------------------------------------------------------------------------------
  // get image tensors & labels
  // --------------------------------------------------------------------------------
  for (let i = 1; i<31; i++){
    img = capture(i.toString())
    imageTensors.push(img)
    //img.dispose()
  
    if (i > 0 && i < 11 ){
      imgLabels.push(1)
    } else if (i > 10 && i < 21 ) {
      imgLabels.push(2)
    } else if (i > 20 && i < 31 ) {
      imgLabels.push(3)
    } else {
      console.log("Loop incorrect")
    }
  }
  // --------------------------------------------------------------------------------
  // randomize imgs & labels
  // --------------------------------------------------------------------------------
  /*
  var i=0, len= imageTensors.length, next, order=[];
  while(i<len)order[i]= ++i; //[1,2,3...]
  order.sort(function(){return Math.random()-.5});
  
  for(i= 0; i<len; i++){
    next= order[i];
    imageTensors.push(imageTensors[next]);
    imgLabels.push(imgLabels[next]);
  }
  imageTensors.splice(1, len);
  imgLabels.splice(1, len);
  */
  // --------------------------------------------------------------------------------
  console.log("train data loaded.");
  
  // --------------------------------------------------------------------------------
  // Initialize
  // --------------------------------------------------------------------------------
  console.log('Loading mobilenet..');
  //async function loadMobilenet() {
  //  return await mobilenet.load();
  //}
  //net = loadMobilenet();
  net = await mobilenet.load()
  const classifier = knnClassifier.create();
  console.log('Successfully loaded model');
  
  // --------------------------------------------------------------------------------
  // train PIZZAS, BURGERS & COKE CANS
  // --------------------------------------------------------------------------------
  const addExample = async (classId, trainImg) => {
    // Capture an image from the web camera.
    //const img = await webcam.capture();
    const img = trainImg;
  
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(img, 'conv_preds');
  
    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
  
    // Dispose the tensor to release the memory.
    //img.dispose(); <------------------------------------------------------
  };
  
  for (let labelId = 1; labelId<4; labelId++){

    // coke can
    if (labelId == 1){
      for (let j=0; j<11; j++){ 
        trainImage = imageTensors[j]
        addExample(labelId, trainImage)
      }
    }
    // pizza
    else if (labelId == 2){
      for (let j=10; j<21; j++){ 
        trainImage = imageTensors[j]
        addExample(labelId, trainImage)
      }
    }
    // burger
    else if (labelId == 3){
      for (let j=20; j<21; j++){ 
        trainImage = imageTensors[j]
        addExample(labelId, trainImage)
      }
    }
    else {
      console.log("someting is wrong!")
    }
  }

  console.log("model trained")
  // -----------------------------------------------------------------------------
  // webcam test
  // -----------------------------------------------------------------------------
  const webcamElement = document.getElementById('webcam');
  const webcam = await tf.data.webcam(webcamElement);

  while (true) {
    if (classifier.getNumClasses() > 0) {
      //const img = await webcam.capture();
      const img = tf.browser.fromPixels(document.getElementById("1"));

      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['Coke Can', 'Pizza', 'Burger'];
      console.log(classes[result.label])
      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

      // Dispose the tensor to release the memory.
      img.dispose();
    }

    await tf.nextFrame();
  }
  
}

start();

// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------
// HELPER FUNCS
// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------

function capture(imgId) {
  // Reads the image as a Tensor from the <image> element.
  this.picture = document.getElementById(imgId);
  const trainImage = tf.browser.fromPixels(this.picture);

  // Normalize the image between -1 and 1. The image comes in between 0-255,
  // so we divide by 127 and subtract 1.
  const trainim = trainImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));

  return trainim;
};

