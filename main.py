import face_recognition
import cv2

# Load known faces and their names
known_face_encodings = [[-0.13085638,  0.14515725,  0.03154191,  0.0010706 , -0.04128804,
       -0.08786723, -0.01328782, -0.19639651,  0.1355302 , -0.07347557,
        0.29548225, -0.05363329, -0.19862579, -0.18197605,  0.00049605,
        0.19992951, -0.22121803, -0.17961462, -0.05678087,  0.02557497,
        0.11343899, -0.05182119,  0.10457353,  0.03703066, -0.05867403,
       -0.37762156, -0.1315535 , -0.11850659,  0.09167586, -0.0749847 ,
       -0.03650719, -0.00733323, -0.22837842, -0.08377364, -0.00935351,
        0.13713388, -0.02915225, -0.11962119,  0.15960486, -0.0436329 ,
       -0.21282633, -0.01912323,  0.01584898,  0.23859377,  0.20154664,
        0.02340976,  0.06367303, -0.11411019,  0.06701241, -0.11405671,
        0.03192584,  0.09388162,  0.1388166 ,  0.04619925,  0.00140539,
       -0.12902287,  0.03629947,  0.10667004, -0.20173873, -0.01050337,
        0.11970539, -0.05647059, -0.03484024, -0.0637953 ,  0.24049178,
        0.10316934, -0.11300766, -0.16815934,  0.14186105, -0.14426379,
       -0.05380412,  0.07092931, -0.18013   , -0.19484508, -0.30160218,
       -0.01254241,  0.39409375,  0.05728187, -0.16174552,  0.0030838 ,
       -0.03608682,  0.01273279,  0.0880044 ,  0.15757972, -0.0111958 ,
        0.06887902, -0.18766379,  0.01692697,  0.19181919, -0.08691332,
       -0.05828002,  0.18798888, -0.05214692,  0.04898638,  0.01654331,
        0.01117021, -0.01767124,  0.07790435, -0.12639785, -0.01309908,
       -0.00639205, -0.03163394, -0.06872992,  0.0511609 , -0.0469166 ,
        0.0436592 ,  0.02213445,  0.06002751, -0.04222156, -0.08140008,
       -0.12783973, -0.07598493,  0.11190043, -0.22321334,  0.21904762,
        0.22209939,  0.02921043,  0.09910384,  0.08685771,  0.10807511,
        0.00186808, -0.03897406, -0.23358801,  0.00852091,  0.08884288,
       -0.01343691,  0.08726226, -0.00795617]]
known_face_names = ["Victor Mongi"]

# Load images and encode faces
# (populate known_face_encodings and known_face_names)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        # Compare face encoding with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Check if any match is found
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw rectangle around the face and label it
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
video_capture.release()
cv2.destroyAllWindows()
