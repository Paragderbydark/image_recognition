import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load and encode known faces
bill_image = face_recognition.load_image_file("photos/bill_gates.jpg")
bill_encoding = face_recognition.face_encodings(bill_image)[0]

elon_image = face_recognition.load_image_file("photos/elon.jpg")
elon_encoding = face_recognition.face_encodings(elon_image)[0]

mark_image = face_recognition.load_image_file("photos/mark.jpeg")
mark_encoding = face_recognition.face_encodings(mark_image)[0]

# Known face encodings and names
known_face_encoding = [
    bill_encoding,
    elon_encoding,
    mark_encoding,
]

known_face_names = [
    "bill",
    "elon",
    "mark"
]

# Track detected students
student = known_face_names.copy()

# Open CSV file for writing
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    # Capture frame-by-frame
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        # Update attendance
        if name in known_face_names:
            if name in student:
                student.remove(name)
                current_time = datetime.now().strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])
                print(f"Recorded: {name} at {current_time}")

    # Display the resulting frame
    cv2.imshow("Attendance System", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
video_capture.release()
cv2.destroyAllWindows()
f.close()
