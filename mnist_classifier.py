import numpy as np
import keras
import cv2 as cv
import pygame

# Training/Testing Configuration
train = False # True for training (and testing using test data), False for testing using the drawing app.

if (train):
    # Hyperparameters
    batch_size = 128
    validation_proportion = 0.05
    learning_rate = 1e-4
    num_epochs = 20

    def to_one_hot(y, n):
        one_hot = np.zeros(n)
        one_hot[y] = 1
        return one_hot

    # Load the MNIST Data
    (train_data, y_train), (test_data, y_test) = keras.datasets.mnist.load_data()
    
    # Make adjustments to the data
    print("Processing data...")
    
    train_data = np.asarray(np.repeat(train_data, 3, -1), dtype = np.float32).reshape((-1, 28, 28, 3))
    test_data = np.asarray(np.repeat(test_data, 3, -1), dtype = np.float32).reshape((-1, 28, 28, 3))
    
    train_labels = []
    for i in range(len(y_train)):
        train_labels.append(to_one_hot(y_train[i], 10))
    del y_train
    train_labels = np.asarray(train_labels)
    
    test_labels = []
    for i in range(len(y_test)):
        test_labels.append(to_one_hot(y_test[i], 10))
    del y_test
    test_labels = np.asarray(test_labels)
    
    print("Finished processing data.")
    
    data_width, data_height = 28, 28
    # Set aside some of the train data for validation sets
    k = round(len(train_data) * validation_proportion)

    model = keras.Sequential()
    
    # Input layer
    model.add(keras.Input(shape = (data_height, data_width, 3), name = "input"))
    
    # Convolutional Layers
    model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides = (1, 1), name = "conv2d-1"))
    model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "maxpool-1"))
    model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), name = "conv2d-2"))
    model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "maxpool-2"))
    model.add(keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), name = "conv2d-3"))
    model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "maxpool-3"))

    model.add(keras.layers.Flatten())

    # Dense layers
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512, activation = "relu", name = "dense-1"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation = "relu", name = "dense-2"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(128, activation = "relu", name = "dense-3"))

    # Output layer
    model.add(keras.layers.Dense(10, activation = "softmax", name = "predictions"))

    # Compile, Train, And Save Model
    model.compile(optimizer = keras.optimizers.Adam(learning_rate),
                    loss = keras.losses.CategoricalCrossentropy(),
                    metrics = ['accuracy'])

    # Train the model.
    print("Training the model...")
    model.fit(train_data[k:], train_labels[k:],
                validation_data = (train_data[:k], train_labels[:k]), epochs = num_epochs, batch_size = batch_size)
    model.save("mnist_classifier.keras", overwrite = True)
    
    # Test the model
    print("Testing the model...")
    counts = 0
    tests = 10
    model.evaluate(test_data, test_labels, batch_size = batch_size)
    for _ in range(tests):
        i = np.random.randint(0, len(test_data), size = 1)[0]
        img = np.expand_dims(test_data[i], 0)
        actual = np.argmax(test_labels[i])
        
        probs = np.reshape(model(img), -1)
        pred = np.argmax(probs)
        if (actual == pred): counts += 1
        print("Prediction: %d, Actual: %d" % (pred, actual))
        cv.imshow("Test Image", img[0])
        cv.waitKey()
    print("Testing sample had an accuracy of %.3f" % (counts/tests))

else:
    model = keras.models.load_model("mnist_classifier.keras")
    
    # Params
    WIDTH = 320
    HEIGHT = 240
    FPS = 60
    SAMPLE_POINTS = 25
    THICKNESS = 2
    PADDING = 25 # Number of pixels in each direction to pad the image by

    # Setup pygame shit
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MNIST Tester")
    clock = pygame.time.Clock()

    pixels = 255 * np.ones((WIDTH, HEIGHT, 3))
    data_x = []
    data_y = []

    # Event loop
    accept_mouse_input = False
    predicted = False
    running = True
    while running:
        # Poll for Events
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if (not accept_mouse_input):
                    accept_mouse_input = True
                    predicted = False
                    pixels = 255 * np.ones((WIDTH, HEIGHT, 3))
                    data_x = []
                    data_y = []
            if event.type == pygame.MOUSEBUTTONUP:
                if (accept_mouse_input):
                    accept_mouse_input = False
            if event.type == pygame.QUIT:
                running = False
                break

        if accept_mouse_input:
            x, y = pygame.mouse.get_pos()
            # Add extra padding around the center
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if (not (0 <= x + dx < WIDTH and 0 <= y + dy < HEIGHT)): continue
                    pixels[x + dx][y + dy] = (0, 0, 0)
            data_x.append(x)
            data_y.append(y)
        else:
            # Plot the splines
            if (len(data_x) != 0):
                # For determining the appropriate bounding box
                minX, maxX = WIDTH - 1, 0
                minY, maxY = HEIGHT - 1, 0
                
                # Basic linear interpolation
                for i in range(len(data_x) - 1):
                    x0, x1 = data_x[i], data_x[i + 1]
                    y0, y1 = data_y[i], data_y[i + 1]
                    # Update bounds
                    for X in [x0, x1]:
                        minX = min(minX, X)
                        maxX = max(maxX, X)
                    for Y in [y0, y1]:
                        minY = min(minY, Y)
                        maxY = max(maxY, Y)
                    
                    for j in range(0, SAMPLE_POINTS):
                        # Compute the coordinates
                        t = j/(SAMPLE_POINTS-1)
                        x = min(max(0, round((1 - t) * x0 + t * x1)), WIDTH - 1)
                        y = min(max(0, round((1 - t) * y0 + t * y1)), HEIGHT - 1)
                        
                        # Add extra padding around the center
                        for dx in range(-THICKNESS, THICKNESS + 1):
                            for dy in range(-THICKNESS, THICKNESS + 1):
                                if (not (0 <= x + dx < WIDTH and 0 <= y + dy < HEIGHT)): continue
                                # Update bounds
                                minX, maxX = max(0, min(minX, x + dx)), min(WIDTH - 1, max(maxX, x + dx))
                                minY, maxY = max(0, min(minY, y + dy)), min(HEIGHT - 1, max(maxY, y + dy))
                                pixels[x + dx][y + dy] = (0, 0, 0)

                if (not predicted):
                    # Transform the drawn image into something the model can process (28 x 28 px)
                    temp = 255 - pixels[minX:maxX+1, minY:maxY+1]
                    n, m = maxX - minX + 1, maxY - minY + 1
                    padded = np.zeros((m + 2 * PADDING, n + 2 * PADDING, 3))
                    for x in range(n - 1):
                        for y in range(m - 1): padded[PADDING + y][PADDING + x] = temp[x][y]
                    del temp
                    
                    # Resize the image and predict it
                    resized = cv.resize(padded, (28, 28), interpolation = cv.INTER_AREA)
                    for i in range(28):
                        for j in range(28):
                            if (max(resized[i][j]) >= 10):
                                for k in range(3): resized[i][j][k] = 255
                    resized = np.asarray(np.expand_dims(resized, axis = 0), dtype = np.float32)
                    for i in range(28):
                        for j in range(28):
                            if (resized[0][i][j][0] == 255): print("#", end="")
                            else: print(" ", end="")
                        print()
                    probs = np.reshape(model(resized), -1)
                    for i in range(len(probs)): print("%d: %.3f" % (i, probs[i]))
                    
                    if (max(probs) >= 0.8): pred = np.argmax(probs)
                    else: pred = np.random.choice(10, 1, p = probs)[0]
                    
                    # cv.imshow("", resized[0])
                    # cv.waitKey()
                    print("Prediction: %d" % pred)
                    predicted = True
                    
                    # cv.imshow("Resized", resized)
                    # cv.waitKey()
                
                # Draw the bounding box.
                for x in range(minX, maxX + 1):
                    pixels[x][minY] = (255, 0, 0)
                    pixels[x][maxY] = (255, 0, 0)
                for y in range(minY, maxY + 1):
                    pixels[minX][y] = (255, 0, 0)
                    pixels[maxX][y] = (255, 0, 0)
    
        # Render changes to the screen
        surface = pygame.surfarray.make_surface(pixels)
        window.blit(surface, (0, 0))
        pygame.display.update()
        clock.tick(FPS)

    # Exit the window.
    pygame.quit()