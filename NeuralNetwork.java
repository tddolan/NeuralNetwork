
public class NeuralNetwork {
  double[] inL, hidL, outL, hidB, outB, hidBGrad, outBGrad;
  double[][] hidW, outW, hidWGrad, outWGrad;
  
  //Create a new neural network with layer sizes x, y, and z
  public NeuralNetwork(int x, int y, int z) {
    inL = new double[x];
    hidL = new double[y];
    outL = new double[z];
    hidB = new double[y];
    hidBGrad = new double[y];
    outB = new double[z];
    outBGrad = new double[z];
    hidW = new double[x][y];
    hidWGrad = new double[x][y];
    outW = new double[y][z];
    outWGrad = new double[y][z];
    
  }
  
  //assigns each weigh and bias to a random number between -1 and 1, and gradients to 0
  public void initialize() {
    for(int i = 0; i < hidW[0].length;i++) {
      hidB[i] = Math.random()*2-1;
      for(int j = 0; j < hidW.length;j++) {
        hidW[j][i] = Math.random()*2-1;
      }
    }
    for(int i = 0; i < outW[0].length;i++) {
      outB[i] = Math.random()*2-1;
      for(int j = 0; j < outW.length;j++) {
        outW[j][i] = Math.random()*2-1;
      }
    }
  }
  
  //helper method for sigmoid activation function
  public static double[] sigmoid(double[] a) {
    double[] temp = new double[a.length];
    for(int i = 0; i < a.length; i++) {
      temp[i] = 1/(1+Math.exp(-a[i]));
    }
    return temp;
  }
  
  //helper method for multiplying previous layer outputs with their associated weight values
  public static double[] multiply(double[] a, double[][] b) {
    double[] temp = new double[b[0].length];
    for(int i = 0; i < b[0].length; i++) {
      double sum = 0;
      for(int j = 0; j < a.length; j++) {
        sum += a[j] * b[j][i];
      }
      temp[i] = sum;
    }
    return temp;
  }
  
  //helper method for adding biases
  public static double[] add(double[] a, double[] b) {
    double[] temp = new double[a.length];
    for(int i = 0; i < a.length; i++) {
      temp[i] = a[i]+b[i];
    }
    return temp;
  }
  
  //helper method for calculating the activation function derivative
  public static double dSigmoid(double a) {
    return a * (1 - a);
  }
  
  // helper method for calculating the cost function derivative
  public static double dCost(double output, double expected) {
    return 2 * (expected - output);
  }
  
  //process an input
  public double[] fwdProp(double[] input) {
    inL = input;
    hidL = multiply(inL, hidW);
    hidL = add(hidL, hidB);
    hidL = sigmoid(hidL);
    
    outL = multiply(hidL, outW);
    outL = add(outL, outB);
    outL = sigmoid(outL);
    return outL;
  }
  
  //add the gradient values for each weight and bias for a single training data
  public void backProp(double[] expected) {
    //propagation for output layer weights and biases
    for (int i = 0; i < outL.length; i++) {
      double dSig = dSigmoid(outL[i]);
      double dCost =dCost(outL[i], expected[i]);
      outBGrad[i] += dSig * dCost;
      for(int j = 0; j < hidL.length; j++) {
        outWGrad[j][i] += hidL[j] * dSig * dCost;
      }
    }
    
    //propagation for hidden layer weights and biases
    for (int i = 0; i < hidL.length; i++) {
      double dSig = dSigmoid(hidL[i]);
      double dCost = 0;
      for(int k = 0; k < outL.length; k++) {
        double dSigOut = dSigmoid(outL[k]);
        double dCostOut =dCost(outL[k], expected[k]);
        dCost += outW[i][k] * dSigOut * dCostOut;
      }
      hidBGrad[i] += dSig * dCost;
      for(int j = 0; j < inL.length; j++) {
        hidWGrad[j][i] += inL[j] * dSig * dCost;
      }
    }
  }
  
  public void applyGrads(double rate) {
    for (int i = 0; i < outL.length; i++) {
      outBGrad[i] += outBGrad[i] * rate;
      for(int j = 0; j < hidL.length; j++) {
        outW[j][i] += outWGrad[j][i] * rate;
        hidB[j] += hidBGrad[j] * rate;
        for(int k = 0; k < inL.length; k++) {
          hidW[k][j] += hidWGrad[k][j] * rate;
        }
      }
    }
    outWGrad = new double[hidL.length][outL.length];
    hidWGrad = new double[inL.length][hidL.length];
    outBGrad = new double[outL.length];
    hidBGrad = new double[hidL.length];
    
  }
  
}
