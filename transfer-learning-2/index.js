// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadTruncatedMobileNet() {
    const mobilenet = await tf.loadLayersModel(
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  
    // Return a model that outputs an internal activation.
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
  }




/**
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
async function getImage() {
    const img = await webcam.capture();
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
  