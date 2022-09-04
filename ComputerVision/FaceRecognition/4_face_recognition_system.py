import face_recognition

# Load the known images
img1 = face_recognition.load_image_file('person_1.jpg')
img2 = face_recognition.load_image_file('person_2.jpg')
img3 = face_recognition.load_image_file('person_3.jpg')
img4 = face_recognition.load_image_file('person_4.jpg')

# Get the face encoding of each person. This can fail if no one is found in the photo.
encode1 = face_recognition.face_encodings(img1)[0]
encode2 = face_recognition.face_encodings(img2)[0]
encode3 = face_recognition.face_encodings(img3)[0]
encode4 = face_recognition.face_encodings(img4)[0]

# Create a list of all known face encodings
known_face_encodings = [
    encode1, encode2, encode3, encode4
]

# Load the image we want to check
unknown_image = face_recognition.load_image_file('unknown_9.jpg')

# Get face encodings for any people in the picture
unknown_face_encodings = face_recognition.face_encodings(unknown_image)

# print(len(unknown_face_encodings))
# There might be more than one person in the photo, so we need to loop over each face we found
for unknown_face_encoding in unknown_face_encodings:

    # Test if this unknown face encoding matches any of the three people we know
    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.6)

    name = "Unknown"

    if results[0]:
        name = "Person 1"
    elif results[1]:
        name = "Person 2"
    elif results[2]:
        name = "Person 3"
    elif results[3]:
        name = "Person 4"

    print(f"Found {name} in the photo!")
