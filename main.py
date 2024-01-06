import face_recognition
import os, sys
import cv2
import numpy as np
import math
import time
from google.cloud import storage
import io
import requests
import json
from PIL import Image


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/vSAUSE/PycharmProjects/pythonProject/pacific-engine-406208-e5350ed64f4f.json'


def face_confidence(face_distance, face_match_threshold=0.6):  // face confidence calculations 
   range = (1.0 - face_match_threshold)
   linear_val = (1.0 - face_distance) / (range * 2.0)


   if face_distance > face_match_threshold:
       return str(round(linear_val * 100, 2)) + '%'
   else:
       value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
       return str(round(value, 2)) + '%'


class FaceRecognition:
   face_locations = []
   face_encodings = []
   face_names = []
   known_face_encodings = []
   known_face_names = []
   process_current_frame = True
   frame_counter = 0
   face_id_counter = 0
   frame_save_counter = 0
   save_interval = 10 #measured in half the seconds eg. 10 = 5 seconds.


   def __init__(self, bucket_name):
       self.bucket_name = bucket_name
       self.storage_client = storage.Client()


   def load_json_file(self, file_path):
       with open(file_path, 'r', encoding='utf-8') as file:
           return json.load(file)


   def download_image(self, url):
       response = requests.get(url)
       image = np.array(bytearray(response.content), dtype=np.uint8)
       return cv2.imdecode(image, cv2.IMREAD_COLOR)


   def upload_to_gcs(self, image, destination_blob_name):  //uploads saved images of faces to google cloud storage
       bucket = self.storage_client.bucket(self.bucket_name)
       blob = bucket.blob(destination_blob_name)


       _, buffer = cv2.imencode('.jpg', image)
       image_bytes = io.BytesIO(buffer)


       blob.upload_from_file(image_bytes, content_type='image/jpeg')
       print(f"Uploaded image to {destination_blob_name}")




   def save_face_image(self, frame, location, recognized_name):  //saves face image snippet from camera
       top, right, bottom, left = location
       top *= 2
       right *= 2
       bottom *= 2
       left *= 2


       face_frame = frame[top:bottom, left:right]


       if face_frame.size > 0:
           subfolder_name = recognized_name if recognized_name != 'Unknown Identity' else f'Unknown_{self.face_id_counter}'
           subfolder_path = os.path.join('captured_faces', subfolder_name)
           destination_blob_name = f'{subfolder_name}/{subfolder_name}_{self.face_id_counter}.jpg'


           self.upload_to_gcs(face_frame, destination_blob_name)


           self.face_id_counter += 1
       else:
           print(f"Skipped saving face")


   def download_and_resize_image(self, url, max_size=(800, 800)):  //downloads all faces from set database and resizes image to encode faster
       response = requests.get(url)
       image = Image.open(io.BytesIO(response.content))
       image.thumbnail(max_size)
       return np.array(image)


   def encode_faces(self, json_file_path):  //encodes the faces from the given dataset before live camera feed begins
       data = self.load_json_file(json_file_path)
       for post in data:
           # Get the first username from the taggedUsers list, if available
           username = post['taggedUsers'][0]['username'] if 'taggedUsers' in post and post[
               'taggedUsers'] else 'Unknown'
           for image_url in post['images']:
               try:
                   image = self.download_and_resize_image(image_url)
                   face_encodings = face_recognition.face_encodings(image)
                   if face_encodings:
                       self.known_face_encodings.extend(face_encodings)  # Add all found encodings
                       self.known_face_names.extend(
                           [username] * len(face_encodings))  # Associate each face with the username
               except Exception as e:
                   print(f"Error processing image {image_url}: {e}")




   def run_recognition(self):  //runs the camera and begins the facial recognition
       video_capture = cv2.VideoCapture(0)
       video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
       video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)


       start_time = time.time()


       if not video_capture.isOpened():
           sys.exit('Video source not found!! Please check if app can use')


       face_id = 0
       while True:
           ret, frame = video_capture.read()


           self.frame_counter += 1
           fps = self.frame_counter / (time.time() - start_time)
           cv2.putText(frame, f"FPS: {fps:.2f}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2, cv2.LINE_AA)


           if self.process_current_frame:
               small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
               rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
               self.face_locations = face_recognition.face_locations(rgb_small_frame)
               self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)


               self.face_names = []

               for face_encoding, location in zip(self.face_encodings, self.face_locations):
                   # Calculate the face distance between the webcam face and known faces
                   face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                   # Check if we have a match
                   best_match_index = np.argmin(face_distances)
                   if face_distances[best_match_index] < 0.426:  # Stricter threshold, can be adjusted, 0.3 giving false negatives, 0.35 giving false negatives, 0.4 giving false negatives, 0.5 false positive, 0.42 prob best, still giving both false pos / neg
                       name = self.known_face_names[best_match_index]
                       confidence = face_confidence(face_distances[best_match_index])
                   else:
                       name = 'Unknown Identity'
                       confidence = 'Unknown'

                   self.face_names.append(f'{name} ({confidence})')


                   self.frame_save_counter +=1


                   if self.frame_save_counter % self.save_interval == 0:
                       self.save_face_image(frame, location, name)


           self.process_current_frame = not self.process_current_frame


           for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
               top *= 2
               right *= 2
               bottom *= 2
               left *= 2


               cv2.rectangle(frame, (left, top), (right, bottom), (0,255,), 2)
               cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), 1)
               cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)


           cv2.imshow('Face Recognition', frame)


           if cv2.waitKey(1) == ord('q'):
               break


       video_capture.release()
       cv2.destroyAllWindows()



if __name__ == '__main__':
   fr = FaceRecognition('facebucketfacial')
   fr.encode_faces('C:/Users/newData200.json')
   fr.run_recognition()
