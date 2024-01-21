using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;

// Artificial Neural Network
public class ANN 
{
    Dictionary<string, int> activationFunctions = new Dictionary<string, int>() {
        {"Sigmoid", 0},
        {"Tanh", 1},
        {"ReLU", 2},
        {"LeakyReLU", 3},
        {"BinaryStep", 4},
        {"Linear", 5},
        {"Input", 6}
    };
    int[] activationFunction;
    float[][] neurons;
    float[][] biases;
    float[][][] weights;
    int[] layers;
    public ANN(int[] layers, string[] activationFunction) {
        this.layers = layers;
        this.activationFunction = new int[activationFunction.Length];
        for (int i = 0; i < activationFunction.Length; i++) {
            try {
                this.activationFunction[i] = activationFunctions[activationFunction[i]];
            } catch (KeyNotFoundException) {
                Debug.LogError($"Activation function {activationFunction[i]} not found.");
            }
        }
        InitNeurons(layers);
        InitBiases(layers);
        InitWeights(layers);
    }
    private void InitNeurons(int[] layers) {
        List<float[]> neuronsList = new List<float[]>();
        for (int i = 0; i < layers.Length; i++) {
            neuronsList.Add(new float[layers[i]]);
        }
        neurons = neuronsList.ToArray();
        biases = neuronsList.ToArray();
    }
    private void InitBiases(int[] layers) {
        biases = new float[neurons.Length][];
        for (int i = 0; i < neurons.Length; i++) {
            biases[i] = new float[neurons[i].Length];
            for (int j = 0; j < neurons[i].Length; j++) {
                biases[i][j] = UnityEngine.Random.Range(-1f, 1f);
            }
        }
    }
    private void InitWeights(int[] layers) {
        List<float[][]> weightsList = new List<float[][]>();
        for (int i = 1; i < layers.Length; i++) {
            List<float[]> layerWeightsList = new List<float[]>();
            int neuronsInPreviousLayer = layers[i-1];
            float xavierInit = Mathf.Sqrt(6f / (neuronsInPreviousLayer + layers[i]));
            for (int j = 0; j < layers[i]; j++) {
                float[] neuronWeights = new float[neuronsInPreviousLayer];
                for (int k = 0; k < neuronsInPreviousLayer; k++) {
                    neuronWeights[k] = UnityEngine.Random.Range(-xavierInit, xavierInit);
                }
                layerWeightsList.Add(neuronWeights);
            }
            weightsList.Add(layerWeightsList.ToArray());
        }
        weights = weightsList.ToArray();
    }

    private float activation(int n, float x) {
        switch(n) {
            case 0:
                return 1/(1+Mathf.Exp(-x));
            case 1:
                return (float)System.Math.Tanh(x);
            case 2:
                return Mathf.Max(0, x);
            case 3:
                return Mathf.Max(0.01f*x, x);
            case 4:
                return x > 0 ? 1 : 0;
            case 5:
            case 6:
            default:
                return x;
        }
    }
    private float activationDerivative(int n, float x) {
        switch(n) {
            case 0:
                return activation(n, x) * (1 - activation(n, x));
            case 1:
                return 1 - Mathf.Pow(activation(n, x), 2);
            case 2:
                return x > 0 ? 1 : 0;
            case 3:
                return x > 0 ? 1 : 0.01f;
            case 4:
                return 0;
            case 5:
            case 6:
            default:
                return 1;
        }
    }
    public float[] FeedForward(float[] inputs) {
        if (inputs.Length != neurons[0].Length) {
            Debug.LogError($"Input length {inputs.Length} does not match input layer length {neurons[0].Length}.");
            return null;
        }
        for (int i = 0; i < inputs.Length; i++) {
            neurons[0][i] = inputs[i];
        }
        for (int i = 1; i < neurons.Length; i++) {
            for (int j = 0; j < neurons[i].Length; j++) {
                float value = 0;
                for (int k = 0; k < neurons[i-1].Length; k++) {
                    value += neurons[i-1][k] * weights[i-1][j][k];
                }
                value += biases[i][j];
                neurons[i][j] = activation(activationFunction[i], value);
            }
        }
        return neurons[neurons.Length-1];
    }
    public void BackPropogate(float[] expectedOutputs, float learningRate) {
        if (expectedOutputs.Length != neurons[neurons.Length-1].Length) {
            Debug.LogError($"Output length {expectedOutputs.Length} does not match output layer length {neurons[neurons.Length-1].Length}.");
            return;
        }
        float[][] errors = new float[neurons.Length][];
        for (int i = 0; i < errors.Length; i++) {
            errors[i] = new float[neurons[i].Length];
        }
        for (int i = 0; i < neurons[neurons.Length-1].Length; i++) {
            errors[neurons.Length-1][i] = (expectedOutputs[i] - neurons[neurons.Length-1][i]) * activationDerivative(activationFunction[activationFunction.Length-1], neurons[neurons.Length-1][i]);
        }
        for (int i = neurons.Length-2; i > 0; i--) {
            for (int j = 0; j < neurons[i].Length; j++) {
                float error = 0;
                for (int k = 0; k < neurons[i+1].Length; k++) {
                    error += errors[i+1][k] * weights[i][k][j];
                }
                errors[i][j] = error * activationDerivative(activationFunction[i], neurons[i][j]);
            }
        }
        for (int i = 0; i < weights.Length; i++) {
            for (int j = 0; j < weights[i].Length; j++) {
                for (int k = 0; k < weights[i][j].Length; k++) {
                    weights[i][j][k] += neurons[i][k] * errors[i+1][j] * learningRate;
                }
            }
        }
        for (int i = 1; i < biases.Length; i++) {
            for (int j = 0; j < biases[i].Length; j++) {
                biases[i][j] += errors[i][j] * learningRate;
            }
        }
    }
    public void Learn(float[][] inputs, float[][] expectedOutputs, int epochs, float learningRate, int batchSize)
    {
        if (inputs.Length != expectedOutputs.Length)
        {
            Debug.LogError($"Number of inputs {inputs.Length} does not match number of outputs {expectedOutputs.Length}.");
            return;
        }
        int totalSamples = inputs.Length;

        int numBatches = totalSamples / batchSize;
        if (totalSamples % batchSize != 0)
        {
            numBatches += 1;
        }

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            int[] indices = new int[totalSamples];
            for (int i = 0; i < totalSamples; i++)
            {
                indices[i] = i;
            }

            System.Random rnd = new System.Random();
            for (int i = totalSamples - 1; i > 0; i--)
            {
                int randomIndex = rnd.Next(0, i + 1);
                int temp = indices[i];
                indices[i] = indices[randomIndex];
                indices[randomIndex] = temp;
            }

            for (int batch = 0; batch < numBatches; batch++)
            {
                int startIndex = batch * batchSize;
                int endIndex = Mathf.Min(startIndex + batchSize, totalSamples);

                for (int i = startIndex; i < endIndex; i++)
                {
                    int dataIndex = indices[i];
                    FeedForward(inputs[dataIndex]);
                    BackPropogate(expectedOutputs[dataIndex], learningRate);
                }
            }
        }
    }
    public float[] OutputSoftmax(float[] inputs) {
        if (inputs.Length != neurons[0].Length) {
            Debug.LogError($"Input length {inputs.Length} does not match input layer length {neurons[0].Length}.");
            return null;
        }
        float[] output = FeedForward(inputs);
        float sum = 0;
        for (int i = 0; i < output.Length; i++) {
            output[i] = Mathf.Exp(neurons[neurons.Length-1][i]);
            sum += output[i];
        }
        for (int i = 0; i < output.Length; i++) {
            output[i] /= sum;
        }
        return output;
    }
    public void Mutate(float mutateRate, float mutateAmount) {
        for (int i = 0; i < weights.Length; i++) {
            for (int j = 0; j < weights[i].Length; j++) {
                for (int k = 0; k < weights[i][j].Length; k++) {
                    if (UnityEngine.Random.Range(0f, 1f) < mutateRate) {
                        weights[i][j][k] += UnityEngine.Random.Range(-mutateAmount, mutateAmount);
                    }
                }
            }
        }
        for (int i = 0; i < biases.Length; i++) {
            for (int j = 0; j < biases[i].Length; j++) {
                if (UnityEngine.Random.Range(0f, 1f) < mutateRate) {
                    biases[i][j] += UnityEngine.Random.Range(-mutateAmount, mutateAmount);
                }
            }
        }
    }

    public ANN(ANNData annData, int[] layers, string[] activationFunction)
    {
        if (layers.Length != annData.layers.Length)
        {
            Debug.LogError($"Number of layers {layers.Length} does not match number of activation functions {activationFunction.Length}.");
            return;
        }
        for (int i = 0; i < layers.Length; i++)
        {
            if (layers[i] != annData.layers[i])
            {
                Debug.LogError($"Layer {i} has {layers[i]} neurons, but loaded ANN has {annData.layers[i]} neurons.");
                return;
            }
        }
        this.activationFunction = new int[activationFunction.Length];
        for (int i = 0; i < activationFunction.Length; i++)
        {
            this.activationFunction[i] = activationFunctions[activationFunction[i]];
        }
        InitNeurons(layers);
        biases = annData.biases;
        weights = annData.weights;
    }

    [Serializable]
    public class ANNData
    {
        public float[][] biases;
        public float[][][] weights;
        public int[] layers;

        public ANNData(ANN ann)
        {
            biases = ann.biases;
            weights = ann.weights;
            layers = ann.layers;
        }
    }

    public void Save(string filePath)
    {
        ANNData annData = new ANNData(this);
        BinaryFormatter bf = new BinaryFormatter();
        FileStream file = File.Create(filePath);
        bf.Serialize(file, annData);
        file.Close();
    }

    public static ANN Load(string filePath, int[] layers, string[] activationFunctions)
    {
        if (File.Exists(filePath))
        {
            BinaryFormatter bf = new BinaryFormatter();
            FileStream file = File.Open(filePath, FileMode.Open);
            ANNData annData = (ANNData)bf.Deserialize(file);
            file.Close();

            return new ANN(annData, layers, activationFunctions);
        }
        else
        {
            Debug.LogError("File not found: " + filePath);
            return null;
        }
    }

    public bool CheckIfSizeSame(int[] layers)
    {
        if (layers.Length != neurons.Length)
        {
            return false;
        }
        for (int i = 0; i < layers.Length; i++)
        {
            if (layers[i] != neurons[i].Length)
            {
                return false;
            }
        }
        return true;
    }
    

}