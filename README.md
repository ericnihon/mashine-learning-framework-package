# Machine Learning Framework Package Documentation


Documentation about the functions and the scope of the machine learning framework 'uenn' created by Uli Binder and Eric Dienhart.

**How to install**

Follow these steps in a console of your choice:
1. Change with the `cd` command into the packages' repository
2. Enter the command `python setup.py install`






(The functional sequence of creating and training a neural network with the package can be viewed in the file `main.py`.)



<details><summary>Components of the 'uenn' framework:</summary>


1. Neural Network


2. Cost Functions

    - MSE / Mean Squared Error

    - CCE / Categorical Cross Entropy (with Softmax)


3. Optimizer

    - Gradient Descent

    - Gradient Descent Momentum


4. PCA

    - Eigenvector Matrix


5. Activation Methods

    - Sigmoid

    - Tanh

    - Relu

    - Leakyrelu


6. DataTools

    - Import

    - Standard Scaling

    - Normal Scaling

    - Datenset Splitting


7. Performance

    - F1 Value Calculator


8. Plotter

    - Error-Plots

    - F1-Value-Plot


9. Saver

    - Saving Plots

    - Saving Thetas


</details>




<details><summary>Methods and Functions</summary>


**uenn.DataTools.importing(path, ground_truth='ground_truth', mnist=True)**

To be able to use the framework, data must first be read in.

        path : str

            The path from which the data is to be read in must be specified.

        ground_truth : str, default 'ground_truth'

            Name of the column containing the ground truth data.

        mnist : bool, default False

            The scikit-learn data set 'MNIST' is automatically read in if the variable is set to True.

**uenn.DataTools.scaling(x, scaler='standardized')**

The function offers to scale the data. This is strongly recommended before the calculation.

        x : object

            Data set to be scaled (without the ground truth).

        scaler : {'standardized', 'normalized'}, default 'standardized'

            Specification of the type of scaling.

**uenn.DataTools.preparing(x, y, ratio)**

To check the results, a certain proportion of the data must be separated in advance.

        x : object

            Data set of the values for the training.

        y : object

            Data set of the ground truth.

        ratio : float, default 0.7

            Specifies the proportion of data that is included in the training set. Specification in decimal / 0. values.

**uenn.PCA.pca(x, cutoff=0.99, path=False)**

The Principal Component Analysis can be used to prepare the data.

        x : object
        
            Data set for the training.

        cutoff : float, default 0.99

            Determines the value of the components to which they still contribute.

        path : str

            Specifies the path where the eigenvector matrix is to be saved.

**uenn.Neuralnet.NN(input_size, layer_number, layer_size, output_size, cost_function='CCE')**

Function for creating the neural network.

        input_size : int

            Specifies the number of features used.

        layer_number : int

            Specifies the number of hidden layers.

        layer_size : int

            Specifies the number of neurons in the hidden layer.

        output_size : int 

            Specifies the number of desired result categories.

        cost_function :  {'CCE', 'MSE'}, default 'CCE'

            The cost_function can be selected here.


**uenn.Neuralnet.train(x, y, epochs, alpha, activation_function='sigmoid', batch=False, beta=False)**

This function is used to enter the (hyper) parameters for calculating the data.

        x : object

            Data set for the training.

        y : object

            Data set for the validation.

        epochs : int
        
            Number of epochs for training.

        alpha : float

            Alpha is required for calculation in training.

        activation_function : {'sigmoid', 'leakyrelu', 'relu', 'tanh'}, default 'sigmoid'

            The activation function can be selected here.

        batch : int

            Specifies the size of a batch.

        beta : float, default 0.75

            Specifies the beta weight if momentum is to be used.

**uenn.Saver.save_plots(plot, path=os.getcwd(), name='plot', bbox_inches='tight', dpi=150)**

This function can be used to create plots and save them for later.

        plot : object

            Data to be displayed in the plot.

        path : str, default os.getcwd()

            Details of the path where the plots are to be saved.

        name : str, default 'plot'

            Specifies the name under which the plot is to be saved.

        bbox_inches : str, default 'tight'

            Enables a better arrangement of the plot on the image to be saved.

        dpi : int, default 150

            Specifies the resolution of the plot.

**uenn.Saver.save_thetas(thetas, path=os.getcwd(), name='thetas')**

This function can be used to save the generated thetas.

        thetas : object

            Thetas which were calculated from the network.

        path : str, default os.getcwd()

            Specifies the path where the thetas are to be saved.

        name : str, default 'thetas'

            Specifies the name under which the thetas are to be saved.


</details>

