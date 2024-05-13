import os
import face_recognition

# Function to load and encode faces from a directory
def load_faces(directory):
    face_encodings = {}
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)
            if len(encoding) > 0:
                face_encodings[filename.split('.')[0]] = encoding[0]
    return face_encodings

# Directory containing images of each person
data_directory = "images"

# Load and encode faces for each person
face_encodings = {}
for person in os.listdir(data_directory):
    if os.path.isdir(os.path.join(data_directory, person)):
        face_encodings[person] = load_faces(os.path.join(data_directory, person))

# Print the number of faces loaded for each person
for person, encodings in face_encodings.items():
    print("Loaded {} face(s) for {}.".format(len(encodings), person))

print(face_encodings)