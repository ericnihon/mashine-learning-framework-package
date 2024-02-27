# Machine Learning Framework Package Documentation


Dokumentation über die Funktionen und den Umfang des selbst erstellten Machine Learning Frameworks 'uenn'.


(Im File `main.py` ist der funktionelle Ablauf des Packages einsehbar.)



<details><summary>Komponenten des 'uenn' Frameworks:</summary>


1. Neuronales Netzwerk


2. Cost Functions

    - MSE / Mean Squared Error

    - CCE / Categorical Cross Entropy (with Softmax)


3. Optimizer

    - Gradient Descent

    - Gradient Descent Momentum


4. PCA

    - Eigenvektormatrix



5. Activation Methods

    - Sigmoid

    - Tanh

    - Relu

    - Leakyrelu

6. DataTools

    - Import

    - Standard-Skalierung

    - Normal-Skalierung

    - Datenset Splittung

7. Performance

    - F1 Wert Berechner


8. Plotter

    - Error-Plots

    - F1-Wert-Plot

9. Saver

    - Plot Speicherung

    - Thetas Speicherung


</details>




<details><summary>Methoden und Funktionen</summary>


**uenn.DataTools.importing(path, ground_truth='ground_truth', mnist=True)**

Um das Framework nutzen zu können, müssen zuerst Daten eingelesen werden.

        path : str

            Der Pfad von dem die Daten eingelesen werden sollen muss angegeben werden.

        ground_truth : str, default 'ground_truth'

            Name der Spalte, in welcher die Ground Truth Daten vorliegen.

        mnist : bool, default False

            Der Scikit-learn Datensatz 'MNIST' wird automatisch eingelesen, wenn die Variable auf True gesetzt wird.

**uenn.DataTools.scaling(x, scaler='standardized')**

Die Funktion bietet die Skalierung der Daten an. Dies wird vor der Berechnung dringend empfohlen.

        x : object

            Zu skalierender Datensatz (ohne der Ground Truth).

        scaler : {'standardized', 'normalized'}, default 'standardized'

            Angabe zur Art der Skalierung.

**uenn.DataTools.preparing(x, y, ratio)**

Zur Überprüfung der Ergebnisse muss ein bestimmter Anteil der Daten im vorhinein abgetrennt werden.

        x : object

            Datasatz der Werte für das Training.

        y : object

            Datasatz der Ground Truth.

        ratio : float, default 0.7

            Angabe zum Anteil der Daten die in den Traingssatz kommen. Angabe in 0. Werten.

**uenn.PCA.pca(x, cutoff=0.99, path=False)**

Die Principal Component Analysis kann zur Aufbereitung der Daten genutzt werden.

        x : object
        
            Datasatz für das Training.

        cutoff : float, default 0.99

            Bestimmt den Wert der Componenten, zu welchem Anteil sie noch beitragen.

        path : str

            Angabe zum Pfad an dem die Eigenvektormatrix abgespeichert werden sollen.

**uenn.Neuralnet.NN(input_size, layer_number, layer_size, output_size, cost_function='CCE')**

Funktion zum Erstellung des Neuronalen Netzwerks.

        input_size : int

            Angabe der Menge an verwendeten Features.

        layer_number : int

            Angabe zur Anzahl der Hidden Layers.

        layer_size : int

            Angabe zur Anzahl der Neuronen im Hidden Layer.

        output_size : int 

            Angabe zur Anzahl der gewünschten Ergebnisskategorien.

        cost_function :  {'CCE', 'MSE'}, default 'CCE'

            Hier kann die Cost_function ausgewählt werden.


**uenn.Neuralnet.train(x, y, epochs, alpha, activation_function='sigmoid', batch=False, beta=False)**

Mit dieser Funktion werden die (Hyper-) Parameter zur Berechnung der Daten eingegeben.

        x : object

            Datasatz für das Training.

        y : object

            Datasatz für die Validation.

        epochs : int
        
            Anzahl an Epochen zum Trainieren.

        alpha : float

            Alpha wird zur Berechnung im Training benötigt.

        activation_function : {'sigmoid', 'leakyrelu', 'relu', 'tanh'}, default 'sigmoid'

            Hier kann die Aktivierungsfunktion ausgewählt werden.

        batch : int

            Angabe zur Größe der Batch-size.

        beta : float, default 0.75

            Angabe zum Beta-Gewicht, wenn Momentum genutzt werden soll.

**uenn.Saver.save_plots(plot, path=os.getcwd(), name='plot', bbox_inches='tight', dpi=150)**

Mit dieser Funktion können Plots erstellt und für später gespeichert werden.

        plot : object

            Daten welche im Plot abgebildet werden sollen.

        path : str, default os.getcwd()

            Angaben zum Pfad an dem die Plots abgespeichert werden sollen.

        name : str, default 'plot'

            Angabe zum Namen unter dem der Plot abgespeichert werden soll.

        bbox_inches : str, default 'tight'

            Ermöglicht eine besser Anordnung des Plots auf dem abzuspeichernden Bild.

        dpi : int, default 150

            Angabe zur Auflösung des Plots.

**uenn.Saver.save_thetas(thetas, path=os.getcwd(), name='thetas')**

Mit dieser Funktion können die erzeugten Thetas abgespeichert werden.

        thetas : object

            Thetas welche aus dem Netzwerk berechnet wurden.

        path : str, default os.getcwd()

            Angabe zum Pfad an dem die Thetas abgespeichert werden sollen.

        name : str, default 'thetas'

            Angabe zum Namen unter dem die Thetas abgespeichert werden sollen.


</details>

