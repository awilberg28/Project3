from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module
import math


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim
import torch.nn as nn


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        "*** YOUR CODE HERE ***"
        super(PerceptronModel, self).__init__()
        weight_vector = torch.Tensor(1, dimensions).zero_()
        self.weight = Parameter(weight_vector)
        

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.weight

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        return tensordot(self.weight, x, dims=([1],[1]))

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        if self.run(x) >= 0: return 1
        else: return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            convergence = False

            "*** YOUR CODE HERE ***"
            while(not convergence):
                mistakes=False
                for sample in dataloader:
                    val = sample['x']
                    label = sample['label']
                    magnitude=self.get_prediction(val)
                    if magnitude!=label.item():
                        self.weight+=label*val
                        mistakes = True
                if mistakes==False:convergence=True

class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()
        self.lr = .01 # learning rate
        self.layer1 = Linear(1, 64)  # Input: (batch_size x 1) → (batch_size x 64)
        self.layer2 = Linear(64, 64) # Hidden: (batch_size x 64) → (batch_size x 64)
        self.output_layer = Linear(64, 1)  # Output: (batch_size x 1)
        



    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = relu(self.layer1(x))       # Nonlinear activation after first layer
        x = relu(self.layer2(x))       # Nonlinear activation after second layer
        out = self.output_layer(x)     # No activation at output (regression task)
        return out

    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        predictions = self.forward(x)
        return mse_loss(predictions, y)
 
        

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=0.01)

        with no_grad():
            stop = False

        while not stop:
            total_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                x = batch['x']
                y = batch['label']

                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            if avg_loss < 0.01:
                stop = True


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        self.linear0 = Linear(input_size, 200)
        self.linear1 = Linear(200, output_size)
        


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        return self.linear1(relu(self.linear0(x)))
 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        
        return cross_entropy(self.run(x), y)

    
        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        loader = DataLoader(dataset, batch_size=300, shuffle=True)
        opt = optim.Adam(self.parameters(), lr=0.004)


        for epoch in range(5):
            total_loss = 0
            for batch in loader:
                x = batch['x']
                y = batch['label']
                
                opt.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                opt.step()


class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        # super(LanguageIDModel, self).__init__()
        # "*** YOUR CODE HERE ***"
        super().__init__()
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.hidden_size = 256 

        # Layers
        self.input_layer = Linear(self.num_chars, self.hidden_size)
        self.hidden_layer = Linear(self.hidden_size, self.hidden_size)
        self.char_layer = Linear(self.num_chars, self.hidden_size)
        self.output_layer = Linear(self.hidden_size, len(self.languages))


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = relu(self.input_layer(xs[0]))

        # Process remaining characters
        for x in xs[1:]:
            h = relu(self.char_layer(x) + self.hidden_layer(h))

        # After processing whole word, predict language
        out = self.output_layer(h)
        return out

    
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(xs)
        return cross_entropy(predictions, y.argmax(dim=1))  # Need target labels as class indices

        

    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.0005)

        converged = False
        epochs = 0
        max_epochs = 25  # Allow more epochs
        while not converged and epochs < max_epochs:
            print("epoch: ", epochs)
            total_loss = 0
            total_samples = 0
            for batch in dataloader:
                x = batch['x']  # (batch_size, length, num_chars)
                y = batch['label']  # (batch_size, 5)

                # Need to move dimensions: (length, batch_size, num_chars)
                x = movedim(x, 1, 0)

                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                total_samples += y.size(0)

            avg_loss = total_loss / total_samples
            epochs+=1
            print("avg_loss", avg_loss)
            if avg_loss < 0.08:  # achieved by fiddling with params
                converged = True

        

def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    heightIn, widthIn = input.shape
    heightWeight, widthWeight = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"
    output_height = heightIn - heightWeight + 1
    output_width = widthIn - widthWeight + 1
    Output_Tensor = torch.zeros(output_height, output_width)


    for x in range(widthIn + 1 - widthWeight):
        for y in range(heightIn + 1 - heightWeight):
            patch = input[y: y + heightWeight, x: x + widthWeight]
            Output_Tensor[y,x] = torch.tensordot(patch , weight, dims = 2)

    "*** End Code ***"
    return Output_Tensor



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """
    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """
        self.linear0 = Linear(676, 200)
        self.linear1 = Linear(200, output_size)
        
        
    def run(self, x):
        return self(x)
 
    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """
        return self.linear1(relu(self.linear0(x)))


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        return cross_entropy(self.forward(x),y)

     
        

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        loader = DataLoader(dataset, batch_size=300, shuffle=True)
        opt = optim.Adam(self.parameters(), lr=0.004)


        for epoch in range(5):
            total_loss = 0
            for batch in loader:
                x = batch['x']
                y = batch['label']
                
                opt.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                opt.step()



class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size,layer_size)

        #Masking part of attention layer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
       
        self.layer_size = layer_size


    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have 
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:
    
        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()
        K=self.k_layer(input)
        Q=self.q_layer(input)
        V=self.v_layer(input)

        """YOUR CODE HERE"""
        
        Q_transposed = torch.transpose(Q, -2, -1)
        numerator=matmul(K,Q_transposed)
        M = numerator / math.sqrt(self.layer_size)

        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        answer = matmul(softmax(M,dim=-1),V)

        return answer
     
