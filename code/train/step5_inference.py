import tensorflow as tf
import numpy as np
import os

class tfliteInference:
    def __init__(self, model_path, pb_model_path=None):
        self.model_path = model_path
        if pb_model_path is not None:
            self.convert_to_tflite(pb_model_path)
        # Load the model
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=8)
        # Set model input
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def convert_to_tflite(self, pb_model_path):
        # Convert the model from saved model(.pb) to tflite
        converter = tf.lite.TFLiteConverter.from_saved_model(pb_model_path)
        tflite_model = converter.convert()
        with open(self.model_path, "wb") as f:
            f.write(tflite_model)

    def run(self, inputData):
        # Preprocess the image before sending to the network.
        inputData = np.expand_dims(inputData, axis=0)

        # The actual detection.
        self.interpreter.set_tensor(self.input_details[0]["index"], inputData)
        self.interpreter.invoke()

        # Save the results.
        mesh = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        return mesh
    
    def get_weight(self, data):
        # weight = self.run(data)
        weight = []
        for i in range(data.shape[0]): 
            data_temp = data[i].astype(np.float32)
            # data_temp = data_temp.reshape((32,64,1))
            output = self.run(data_temp).reshape((1,-1))
            weight.extend(output)
            # import pdb; pdb.set_trace()
        return (np.array(weight))


if __name__ == "__main__":
    tflitepath = './output4_6/models/Audio2Face.tflite'
    model_path = './output4_6/models/Audio2Face'

    if os.path.exists(tflitepath):
        model_path = None
    
    inference = tfliteInference(tflitepath,model_path)

    data = np.load(os.path.join('./lpc/1114_2_06.npy'))
    print(data.shape)

    weight = inference.get_weight(data[:20])

    print(weight.shape)

    # weight = []
    # for i in range(100): #data.shape[0]
    #     data_temp = data[i]
    #     data_temp = data_temp.reshape((32,64,1)).astype(np.float32)
    #     output = inference.run(data_temp)
    #     weight.extend(output)
    # weight = np.array(weight)
    # print(weight.shape)

    # # Set numpy unlimited write
    # np.set_printoptions(threshold=np.inf)
    # with open('./output4_6/weight.txt', 'w') as f:
    #     f.write(f"{str(weight)}")
