async function getModifiedMobilenet()
{
  async function freezeModelLayers(trainableLayers,mobilenetModified)
  {
    for (const layer of mobilenetModified.layers) 
    {
      layer.trainable = false;
      for (const tobeTrained of trainableLayers) 
      {
        if (layer.name.indexOf(tobeTrained) === 0) 
        {
          layer.trainable = true;
          break;
        }
      }
    }
    return mobilenetModified;
  }
  
  const trainableLayers = ['denseModified','conv_pw_13_bn','conv_pw_13','conv_dw_13_bn','conv_dw_13'];
  // tf.mobilenet.load() doesnt give layers
  const mobilenet =  await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json')
  console.log('Mobilenet model is loaded');

  const x = mobilenet.getLayer('global_average_pooling2d_1');

  const predictions = tf.layers.dense({units: 2, activation: 'softmax',name: 'denseModified'}).apply(x.output);
  
  let mobilenetModified = tf.model({inputs: mobilenet.input, outputs: predictions, name: 'modelModified' });
  console.log('Mobilenet model is modified');

  mobilenetModified = freezeModelLayers(trainableLayers,mobilenetModified)
  console.log('ModifiedMobilenet model layers are freezed');

  mobilenetModified.compile({loss: 'categoricalCrossentropy', optimizer: tf.train.adam(1e-3), metrics: ['accuracy','crossentropy']});

  return mobilenetModified 
}


async function MNet()
{
  async function freezeModelLayers(trainableLayers,mobilenetModified)
  {
    for (const layer of mobilenetModified.layers) 
    {
      layer.trainable = false;
      for (const tobeTrained of trainableLayers) 
      {
        if (layer.name.indexOf(tobeTrained) === 0) 
        {
          layer.trainable = true;
          break;
        }
      }
    }
    return mobilenetModified;
  }
  
  const net = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json')
  console.log('Loaded!')
  //console.log(net)
  //console.log('getting layers..')
  //net.layers.forEach(layer => console.log(layer.name))
  
  const x = net.getLayer('global_average_pooling2d_1');
  const predictions = tf.layers.dense({units: 2, activation: 'softmax',name: 'denseModified'}).apply(x.output);
  
  let mobilenetModified = tf.model({inputs: net.input, outputs: predictions, name: 'modelModified' });
  console.log('Mobilenet model is modified');
  //mobilenetModified.layers.forEach(layer => console.log(layer.name))
  
  mobilenetModified = freezeModelLayers(['denseModified','conv_pw_13_bn','conv_pw_13','conv_dw_13_bn','conv_dw_13'],mobilenetModified)
  console.log('ModifiedMobilenet model layers are freezed');

  //mobilenetModified.compile({loss: 'categoricalCrossentropy', optimizer: tf.train.adam(1e-3), metrics: ['accuracy']});  

}