__authors__ = ['1666134','1703200','1668438','1672891']
__group__ = '07'

from utils_data import *
import time
from Kmeans import *
from KNN import *
import matplotlib.pyplot as plt


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    #Funció feta servir per saber un progrés aproximat de les execucions
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def retrieval_by_color(images, color_labels, color):
        matching_color=[]
        for i, color_i in enumerate(color_labels):
            if all(c in color_i for c in color):
                matching_color.append(images[i])
        return matching_color

def retrieval_by_shape(images,shape_labels, shape):
        matching_images = []
        for i, label in enumerate(shape_labels):
            if label == shape:
                matching_images.append(images[i])
        return matching_images
    
def retrieval_combined(images, color_labels, shape_labels, color, shape):
        matching_images = []
        for i in range(len(images)):
            shape_i = shape_labels[i]
            color_i = color_labels[i]
            if shape in shape_i:
                if all(color in color_i for color in color):
                    matching_images.append(images[i])
        return matching_images
    
    
def kmean_statistics(kmeans, Kmax, images):
    
    def autolabel(rects):
        """Funcion para agregar una etiqueta con el valor en cada barra"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    k = [i for i in range(2, Kmax + 1)]
    met_values = [0,0]
    num_iterations = [0,0]
    iter_k_b = [[] for i in range(2, Kmax + 1)]
    convergence_times = [0,0]
    
    for K in range(2, Kmax + 1):
        start_time = time.time()
        kmeans.K = K  
        kmeans.fit()
        end_time = time.time()
        num_iter = kmeans.num_iter
        if kmeans.options['fitting'] == 'WCD':
            metode = "WCD"
            met = kmeans.withinClassDistance()
        elif kmeans.options['fitting'] == 'IC':
            metode = "Inter class"
            met = kmeans.inter_class()
        elif kmeans.options['fitting'] == 'FD':
            metode = "Fisher discriminant"
            met = kmeans.fisher_disc()
        convergence_time = end_time - start_time

        met_values.append(round(met))
        num_iterations.append(round(num_iter))
        convergence_times.append(round(convergence_time*1000))

        print(f"K={K}: {metode}={met}, Iterations={num_iter}, Time={convergence_time}")
    
    count = 0
    
    for j in images:
        count += 1
        printProgressBar(count, len(images), prefix = 'Progress:', suffix = 'Complete')
        km_init = kmeans.options['km_init']
        if kmeans.options['fitting'] == "WCD":
            kmeans = KMeans(j, 1, options = {'km_init': km_init, 'fitting': 'WCD'})

        elif kmeans.options['fitting'] == "IC":
            kmeans = KMeans(j, 1, options = {'km_init': km_init, 'fitting': 'WCD'})

        elif kmeans.options['fitting'] == "FD":
            kmeans = KMeans(j, 1, options = {'km_init': km_init, 'fitting': 'WCD'})

        for i in k:
            kmeans.K = i
            kmeans.fit()
            iter_k_b[i - 2].append(kmeans.num_iter)
            
    # Creating a figure and axis object
    fig, ax = plt.subplots(figsize=(16, 9))
    # Setting the bar width
    bar_width = 0.3
    # Setting the position of the bars on the x-axis
    r1 = [i for i in range(0, Kmax+1)]
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    # Creating the bars
    rects1 = ax.bar(r1, num_iterations, color='green', width=bar_width, edgecolor='grey', label='iterations')
    rects2 = ax.bar(r2, convergence_times, color='orange', width=bar_width, edgecolor='grey', label='temps (ms)')
    rects3 = ax.bar(r3, met_values, color='blue', width=bar_width, edgecolor='grey', label=metode)
    # Adding labels
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    ax.set_xlabel('K', fontweight='bold', fontsize=15)
    ax.set_ylabel('Values', fontweight='bold', fontsize=15)
    ax.set_xticks([r + bar_width for r in range(Kmax+1)])
    ax.set_xticklabels(r1)
    # Creating a legend
    ax.legend()
    plt.savefig('Iterations_Time_WCD_vs_K.png')
    # Show the plot
    plt.show()
    
    K = [i for i in range(2, Kmax + 1)]
    
    #Boxplot for iterations
    plt.clf()
    plt.boxplot(iter_k_b)
    plt.xlabel('K')
    plt.ylabel('iteracions')
    plt.xticks([i for i in range(1, Kmax)], K, fontsize=10)
    plt.savefig('Iterations_Boxplot.png')
    plt.show()


def Get_color_accuracy(predicted_labels, ground_truth_labels):
    total_similarity = 0.0
    for pred, gt in zip(predicted_labels, ground_truth_labels):
        pred_set = set(pred)
        gt_set = set(gt)
        total_similarity += len(pred_set.intersection(gt_set)) / len(pred_set.union(gt_set))    # percentatge mitjà de similitud
    accuracy = (total_similarity / len(ground_truth_labels)) * 100.0
    return accuracy

def Get_shape_accuracy(predicted_labels, ground_truth_labels):

    correct_count = sum(1 for pred, gt in zip(predicted_labels, ground_truth_labels) if pred == gt) #zip per recorrer les 2 llistes simultaniament

    accuracy = (correct_count / len(predicted_labels)) * 100.0

    return accuracy

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    all_images = np.concatenate((train_imgs, test_imgs))
    all_colors = np.concatenate((train_color_labels, test_color_labels))
    all_classes = np.concatenate((train_class_labels, test_class_labels))
    
    #TEST FUNCIONS QUALITATIVES / RETRIEVALS
    """
    color=retrieval_by_color(train_imgs,train_color_labels,["Blue", "White"])
    visualize_retrieval(color, 10, info=None, ok=None, title='', query=None)
    shape=retrieval_by_shape(train_imgs,train_class_labels,"Dresses")
    visualize_retrieval(shape, 10, info=None, ok=None, title='', query=None)
    shape_color=retrieval_combined(train_imgs,train_color_labels,train_class_labels,["Blue", "White"],"Dresses")
    visualize_retrieval(shape_color, 10, info=None, ok=None, title='', query=None)
    """
    
    
    #ACCURACY KNN DEFAULT
    
    """
    opt = {'fitting': 'IC'}
    kmeans = KMeans(train_imgs[0], K=2, options = opt)
    Kmax = 13
    kmean_statistics(kmeans, Kmax, train_imgs)
    """
    
    #Estadistic kmeans
    """
    max_k=13
    accuracies = []
    k_values = list(range(1, max_k + 1))
    for k in k_values:
        knn = KNN(train_imgs, train_class_labels)
        knn_labels = knn.predict(test_imgs, k)
        accuracy = Get_shape_accuracy(knn_labels, test_class_labels)
        accuracies.append(accuracy)
    plt.bar(k_values, accuracies, color='orange')
    #plt.text(0.5, 1.08, 'Dataset of 851 images', horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Accuracy vs K in KNN')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    """
    
    #ACCURACY KMEANS DEFAULT
    """
    accuracies_by_k = []
    max_k=13
    for k in range(1, max_k + 1):
        color_predictions = []
        for i, image in enumerate(all_images):
            printProgressBar(i, len(all_images), suffix = 'Complete')
            KM = KMeans(image, K=k)
            KM.fit()
            color_predictions.append(set(get_colors(KM.centroids)))
        
        accuracy = Get_color_accuracy(color_predictions, all_colors)
        accuracies_by_k.append(accuracy)

        print("Color Accuracy for K =", k, ":", accuracy, "%")

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, max_k + 1), accuracies_by_k, color='skyblue', edgecolor='black')
    plt.title('Color Accuracy by K in KMeans', fontsize=16)
    plt.xlabel('K (Number of clusters)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xticks(range(1, max_k + 1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    """
    
    
    #KNN diferents distàncies
    """
    knn = KNN(train_imgs, train_class_labels)
    k = 5
    
    distances = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'minkowski', 'canberra']

    # comparativa d'accuracy 
    shape_accuracy_results = []
    
    for distance in distances:
       knn.get_k_neighbours(test_imgs, k, distance_type=distance)
       predictions = knn.get_class()
       
       shape_accuracy = Get_shape_accuracy(predictions, test_class_labels)
       shape_accuracy_results.append(shape_accuracy)
       

    plt.figure(figsize=(12, 6))
    plt.bar(distances, shape_accuracy_results, color='skyblue')
    plt.xlabel('Distància')
    plt.ylabel('Shape Accuracy')
    plt.title('Shape Accuracy de KNN amb diferents distàncies')
    plt.xticks(rotation=45, ha='right')


    plt.tight_layout()
    plt.show()
    
    #comparativa per temps
    times = []
    
    for distance in distances:
       start_time = time.time()
       knn.get_k_neighbours(test_imgs, k, distance_type=distance)
       predictions = knn.get_class()
       end_time = time.time()
       times.append(end_time - start_time)
      
       

    plt.figure(figsize=(12, 6))
    plt.bar(distances, times, color='skyblue')
    plt.xlabel('Distància')
    plt.ylabel('temps')
    plt.title('temps de KNN amb diferents distàncies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    """
    
    #KMEANS: ACCURACY, TIME, AND ITERATIONS VS K
    """
    all_images = np.concatenate((train_imgs, test_imgs))
    all_colors = np.concatenate((train_color_labels, test_color_labels))
    all_classes = np.concatenate((train_class_labels, test_class_labels))
    
    accuracy = []
    times = []
    iters = []
    for K in range(2, 14):
        color_predictions = []
        for i,image in enumerate(all_images):
            printProgressBar(i, len(all_images), prefix = f'K: {K} out of 14', suffix = 'Complete')
            start = time.time()
            KM = KMeans(image, K=K)
            KM.fit()
            end = time.time()
            color_predictions.append(set(get_colors(KM.centroids)))
        acc = Get_color_accuracy(color_predictions, all_colors )
        accuracy.append(acc)
        times.append(end-start)
        iters.append(KM.num_iter)
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(range(2,14), accuracy)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs K')

    plt.subplot(1,3,2)
    plt.plot(range(2,14), iters)
    plt.xlabel('K')
    plt.ylabel('Iterations')
    plt.title('Number of iteration vs K')

    plt.subplot(1,3,3)
    plt.plot(range(2,14), times)
    plt.xlabel('K')
    plt.ylabel('Time')
    plt.title('Convergence time vs K')

        
    plt.tight_layout()
    plt.show()
    """
    
    #KNN: ACCURACY AND TIME VS K, aquest plot l'hem fet servir com a test però no hem ficat els resultats a l'informe.
    """
    accuracies = [] 
    timing = []
    Knn = KNN(train_imgs, train_class_labels)
    for i,K in enumerate(range(2, 14)):
        printProgressBar(i, 13, prefix = f'K = {K}', suffix = 'Complete')
        start = time.time()
        prediction = Knn.predict(test_data = test_imgs, k=K)
        end = time.time()
        accuracies.append(Get_shape_accuracy(prediction, test_class_labels))
        timing.append(end-start)
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(2,14), accuracies)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs K')

    plt.subplot(1,2,2)
    plt.plot(range(2,14), timing)
    plt.xlabel('K')
    plt.ylabel('Time')
    plt.title('Time vs K')

    plt.tight_layout()
    plt.show()
    """
    
    #KNN ACCURACIES VS K + PLOT
    """
    max_k=13
    accuracies = []
    k_values = list(range(1, max_k + 1))
    print(k_values)
    count = 0
    for k in k_values:
        printProgressBar(count, max_k, suffix = 'Complete')
        knn = KNN(train_imgs, train_class_labels)
        knn_labels = knn.predict(test_imgs, k)
        accuracy = Get_shape_accuracy(knn_labels, test_class_labels)
        accuracies.append(accuracy)
        count+=1

    plt.plot(k_values, accuracies)
    #plt.text(0.5, 1.08, 'Dataset of 851 images', horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Accuracy vs K in KNN')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    """
    
    #KNN DIFERENTS DISTÀNCIES
    """
    knn = KNN(train_imgs, train_class_labels)
    k = 5
    
    distances = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'minkowski', 'canberra', 'hamming']
    # Llistes per emmagatzemar els resultats d'exactitud per a cada distància
    shape_accuracy_results = []
    # Bucle per a cada distància
    
    for distance in distances:
       knn.get_k_neighbours(test_imgs, k, distance_type=distance)
       predictions = knn.get_class()
       
       shape_accuracy = Get_shape_accuracy(predictions, test_class_labels)
       shape_accuracy_results.append(shape_accuracy)

    plt.figure(figsize=(12, 6))
    plt.bar(distances, shape_accuracy_results, color='skyblue')
    plt.xlabel('Distància')
    plt.ylabel('Shape Accuracy')
    plt.title('Shape Accuracy de KNN amb diferents distàncies')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    """
    
    #TEST PLOTS FOR DIFFERENTS METHODS AND LLINDARS FOR THE BEST_K
    """
    heuristics_list = ['WCD', 'IC', 'FD']
    fact = 1
    fig, ax = plt.subplots(3, 2, sharey=True)
    kmins = KMeans(train_imgs[0], 2)
    for k, j in enumerate(heuristics_list): #multiprocessing
        #Hace un KMeans con cada heurística
        opt = {'fitting': j, 'km_init': 'random'}
        kmins._init_options(opt)
        percentage = []
        tim = []
        if j == "IC":
            red = 20 ** 4
            lab = "10^11"
        else:
            red = 100
            lab = "100"
        dist = []
        K = []
        for i in range(0, 11 * fact):
            #Se guarda los valores de tiempo, heurística y k según el umbral usado
            #printProgressBar(i, 11 * fact, suffix = 'Complete', prefix= [j,i])
            percentage.append(i / (10 * fact))
            kmins.p = i / (10 * fact)
            inicio = time.time()
            kmins.find_bestK(13)
            fin = time.time()
            K.append(kmins.K)
            if j == "WCD":
                val = kmins.withinClassDistance()
            elif j == "IC":
                val = kmins.inter_class()
            elif j == "FD":
                val = kmins.fisher_disc()
            #print(val/red)
            #print(fin-inicio)
            dist.append(kmins.val/red)
            tim.append(fin - inicio)
        print('Tim: ', tim)
        print('Dist: ', dist)
        print('K: ', K)
        ax[k][0].set_ylim(0, 14)
        ax[k][0].set_xlabel('percentatge')
        ax[k][0].set_ylabel('temps')
        ax[k][0].plot(percentage, tim)
        ax[k][1].set_ylim(0, 14)
        ax[k][1].set_xlabel('percentatge')
        ax[k][1].set_ylabel('k')
        ax[k][1].plot(percentage, K)
    plt.savefig('best_k.png')
    """
    
    #KMEANS ACCURACIES PER HEURISTIC
    """
    color_accuracies = []  
    max_K=13
    color_accuracy_results = []
    methods=['WCD','IC','FD']
    for method in methods:     
        color_predictions = []
        options={'fitting': str(method)}
        print(options)
        for i,image in enumerate(all_images):       
            printProgressBar(i, len(all_images), prefix = method, suffix = 'Complete')
            KM = KMeans(image, K=5, options=options)
            best_k = KM.find_bestK(max_K)
            KM.fit()
            color_predictions.append(set(get_colors(KM.centroids)))
        color_accuracy_results.append(Get_color_accuracy(color_predictions, all_colors))
    print("--------------------------------------------------------------------")
    print(f"Average Color Accuracy for method {method}: {color_accuracy_results}")
    print("Methods: ",methods)
    print("Accuracies: ", color_accuracy_results)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(methods, color_accuracy_results, color='salmon')
    plt.xlabel('Heurístiques')
    plt.ylabel('Color Accuracy')
    plt.title('Color Accuracy del KMeans amb diferents heurístiques')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    """
    