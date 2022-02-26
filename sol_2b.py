import layers
import numpy as np
import matplotlib.pyplot as plt
import data_generators

class Network(layers.BaseNetwork):
    #TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer,parameters=None):
        # you should always call __init__ first 
        super().__init__()
        #TODO: define your network architecture here
        self.hidden_units= parameters["hidden_units"]
        self.linear = layers.Linear(data_layer,self.hidden_units)
        self.bias = layers.Bias(self.linear)
        self.relu= layers.Relu(self.bias)

        self.linear1= layers.Linear(self.relu,1)
        self.bias1= layers.Bias(self.linear1)
        
        #TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)
        self.set_output_layer(self.bias1)

class Trainer:
    def __init__(self):
        pass
    
    def define_network(self, data_layer, parameters):
        
        hidden_units = parameters["hidden_units"] #needed for prob 2, 3, 4, mnist
        hidden_layers = parameters["hidden_layers"] #needed for prob 3, 4, mnist
        
        #TODO: construct your network here
        network = Network(data_layer, parameters=parameters )
        return network
    
    def setup(self, training_data):
        x, y = training_data
        #store training data
        self.training_data= training_data
        #TODO: define input data layer
        self.data_layer = layers.Data(x)
        #class Data _init__(self, data)
        #TODO: construct the network. you don't have to use define_network.
        parameters={"hidden_units":7,"hidden_layers":1}
        self.network = Network(self.data_layer,parameters)#######################
        #TODO: use the appropriate loss function here
        self.loss_layer = layers.SquareLoss(self.network.get_output_layer(),y)  


        #TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        self.optim =  layers.SGDSolver(0.0058, self.network.get_modules_with_parameters())  ################check

        return self.data_layer, self.network, self.loss_layer, self.optim
    
    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function
        loss= self.loss_layer.forward()
        self.loss_layer.backward()
        self.optim.step()
        return loss

    def get_num_iters_on_public_test(self):
        #TODO: adjust this number to how much iterations you want to train on the public test dataset for this problem.
        return 8000
    
    def train(self, num_iter):
        train_losses = []
        #TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.
        for i in range(num_iter):
            train_losses.append(self.train_step())
        # you have to return train_losses for the function
        return train_losses
    
#DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):

    #setup the trainer
    trainer = Trainer()
    
    #DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.
        dataset = data_generators.data_2b()
        x_train = dataset["train"][0]
        x_test= dataset["test"][0]
        y_train = dataset["train"][1]
        y_test= dataset["test"][1]

        trainer.setup(dataset["train"])
        iter=8000
        loss=trainer.train(iter)
        print(loss[-1])
        iterations = [i for i in range(1,iter+1)]
        plt.title("Loss covergence with iterations")
        plt.xlabel("No. of Iterations")
        plt.ylabel(" Loss ")
        plt.plot(iterations,loss)
        plt.show()

        trainer.data_layer.set_data(dataset["test"][0])
        pred = trainer.network.forward()
        mse =  1/2 * ((pred - y_test)**2).mean()
        print('MSE = ', mse)

        #plt.title("Predicted vs Actual labels")
        #plt.plot(x_test, pred)
        #plt.plot(x_test, y_test)
        #plt.show()
        
    else:
        #DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out

if __name__ == "__main__":
    main()
    pass
