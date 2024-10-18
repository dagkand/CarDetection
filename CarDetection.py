import cv2

cap = cv2.VideoCapture('./videos/highway.mp4')

# Use XML classifier
car_cascade = cv2.CascadeClassifier('haarcascade.xml')

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Read until the video is completed
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect cars in the video (with adjusted parameters)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50), maxSize=(200, 200))

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Car Detection', frame)

    # Press 'Q' on the keyboard to exit or Control C in the terminal :)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()