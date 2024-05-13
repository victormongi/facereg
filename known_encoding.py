import face_recognition

# Load images of known individuals and generate face encodings
known_face_encodings = []
known_face_names = []

# Load images of known individuals
image_paths = ["images_single/victor_mongi.png"]

for path in image_paths:
    # Load image
    image = face_recognition.load_image_file(path)
    
    # Find face locations
    face_locations = face_recognition.face_locations(image)
    
    # Encode the face(s)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Add face encodings and corresponding names to the lists
    known_face_encodings.extend(face_encodings)
    known_face_names.append("Name of person")

print(known_face_encodings)