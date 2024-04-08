import numpy as np
import cv2
from PIL import Image
import os

if __name__ == "__main__":   

    def train_data():
        '''
        Train data
        '''
        # Directory path where the face images are stored.
        path = '/Users/kianmontana/Desktop/Face Regconition/images'
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("\n[INFO] Training...")
        # Haar cascade file for face detection
        detector = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
        
        def getImagesAndLabels(path):
            """
            Load face images and corresponding labels from the given directory path.
        
            Parameters:
                path (str): Directory path containing face images.
        
            Returns:
                list: List of face samples.
                list: List of corresponding labels.
            """
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []
        
            for imagePath in imagePaths:
                # Convert image to grayscale
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                
                # Extract the user ID from the image file name
                id = int(os.path.split(imagePath)[-1].split("-")[1])
        
                # Detect faces in the grayscale image
                faces = detector.detectMultiScale(img_numpy)
        
                for (x, y, w, h) in faces:
                    # Extract face region and append to the samples
                    faceSamples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(id)
        
            return faceSamples, ids
        
        faces, ids = getImagesAndLabels(path)
        
        # Train the recognizer with the face samples and corresponding labels
        recognizer.train(faces, np.array(ids))
        
        # Save the trained model into the current directory
        recognizer.write('trainer.yml')
        
        print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    def create_directory(directory):
        """
        Create a directory if it doesn't exist.
    
        Parameters:
            directory (str): The path of the directory to be created.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Create 'images' directory if it doesn't exist
    create_directory('images')
    
    # Load the pre-trained face cascade classifier
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open a connection to the default camera (camera index 0)
    cam = cv2.VideoCapture(0)
    
    # Set camera dimensions
    cam.set(3, 640)
    cam.set(4, 480)
    
    # Initialize face capture variables
    count = 0
    face_id = input('\nEnter user id (MUST be an integer) and press <return> -->  ')
    print("\n[INFO] Initializing face capture. Look at the camera and wait...")
    
    while True:
        # Read a frame from the camera
        ret, img = cam.read()
    
        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
            # Increment the count for naming the saved images
            count += 1

            # Save the captured image into the 'images' directory
            cv2.imwrite(f"./images/Users-{face_id}-{count}.jpg", gray[y:y+h, x:x+w])
    
            # Display the image with rectangles around faces
            cv2.imshow('image', img)
    
        # Press Escape to end the program
        k = cv2.waitKey(100) & 0xff
        if k < 30:
            break
    
        # Take 30 face samples and stop video. You may increase or decrease the number of
        # images. The more, the better while training the model.
        elif count >= 30:
            break
    
    # Release the camera
    cam.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    train_data()

    
