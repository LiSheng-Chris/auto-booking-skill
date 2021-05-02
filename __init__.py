from mycroft import MycroftSkill, intent_file_handler
from picamera import PiCamera
from time import sleep
import requests
import json
import webbrowser
import datetime

import argparse
import os
# import keras.backend as K
# import cv2
# from keras.models import Model
# from keras.layers import Dense, Input, Lambda, Dot, Conv2D, MaxPool2D, Flatten
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
import random
from google.cloud import storage

# client = storage.Client.from_service_account_json('./static/e-charger-303306-510a928eb8dd.json')

def get_args():
    '''模型建立好之后只需要在这里调参
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default="/home/pi/ISAPM/temp/image.jpg", type=str)  # face_image path
    parser.add_argument("--path", default="lfw-deepfunneled/", type=str) # images lib path
    parser.add_argument("--subset", default=True, type=bool )  # subest indicator 500 face folder for subset
    config = parser.parse_args()
    return config

def identify_face(cfg):

    # # default similarity and folderPath
    # similarity = 0
    # folderPath = ""
    # # Define the path for the downloaded images
    # PATH = cfg.path
    #
    # # We only use subset of this dataset, say 500 folders
    # USE_SUBSET = cfg.subset
    # # USE_SUBSET = False
    #
    # dirs = sorted(os.listdir(PATH))
    #
    # # print(dirs)
    # print("Total number of classes (persons) in the full dataset: ", len(dirs))
    # if USE_SUBSET:
    #     dirs = dirs[:500]
    # print("Total number of classes (persons) in workshop: ", len(dirs))
    #
    # # Build two dictionaries
    # name_to_classid = {d: i for i, d in enumerate(dirs)}
    # # print(name_to_classid)
    # classid_to_name = {v: k for k, v in name_to_classid.items()}
    # # print(classid_to_name)
    #
    # # read all directories
    # img_paths = {c: [PATH + subfolder + "/" + img
    #                  for img in sorted(os.listdir(PATH + subfolder))]
    #              for subfolder, c in name_to_classid.items()}
    # # print(img_paths)
    #
    # # retrieve all images
    # all_images_path = []
    # for img_list in img_paths.values():
    #     all_images_path += img_list
    # # print(all_images_path)
    #
    # # map to integers
    # path_to_imageid = {v: k for k, v in enumerate(all_images_path)}
    # imageid_to_path = {v: k for k, v in path_to_imageid.items()}
    # # print(path_to_imageid)
    # # print(imageid_to_path)
    #
    # # build mappings between images and class
    # classid_to_imageid = {k: [path_to_imageid[path] for path in v] for k, v in img_paths.items()}
    # imageid_to_classid = {v: c for c, imgs in classid_to_imageid.items() for v in imgs}
    #
    #
    # # print(classid_to_imageid)
    # # print(imageid_to_classid)
    #
    #
    # # # Prepare dataset to train a deep learning model for person verification
    # #
    # # - Build paris of positive and negative training images
    # # - Split dataset into train and test subsets for model training
    # #
    #
    # # In[51]:
    #
    #
    # # build pairs of positive image ids for a given classid
    # def build_pos_pairs_for_imageid(classid, max_num=50):
    #     imgs = classid_to_imageid[classid]
    #     if len(imgs) == 1:
    #         return []
    #     pos_pairs = [(imgs[i], imgs[j])
    #                  for i in range(len(imgs))
    #                  for j in range(i + 1, len(imgs))]
    #     random.shuffle(pos_pairs)
    #     return pos_pairs[:max_num]
    #
    #
    # # build pairs of negative image ids for a given classid
    # def build_neg_pairs_for_imageid(classid, classes, max_num=20):
    #     imgs = classid_to_imageid[classid]
    #     neg_classes_imageid = random.sample(classes, max_num + 1)
    #     if classid in neg_classes_imageid:
    #         neg_classes_imageid.remove(classid)
    #     neg_pairs = []
    #     for id2 in range(max_num):
    #         img1 = imgs[random.randint(0, len(imgs) - 1)]
    #         imgs2 = classid_to_imageid[neg_classes_imageid[id2]]
    #         img2 = imgs2[random.randint(0, len(imgs2) - 1)]
    #         neg_pairs += [(img1, img2)]
    #     return neg_pairs
    #
    #
    # def open_all_images(id_to_path):
    #     all_imgs = []
    #     for path in id_to_path.values():
    #         #         print(path)
    #         if not path.endswith('.ini'):
    #             temp = cv2.imread(path)
    #             temp = cv2.resize(temp, (100, 100))
    #         # print("An exception occurred of path: " + path)
    #         all_imgs += [np.expand_dims(temp, 0)]
    #     return np.vstack(all_imgs)
    #
    #
    # def build_train_test_data(num_classes, split=0.8):
    #     listX1 = []
    #     listX2 = []
    #     listY = []
    #     split = int(num_classes * split)
    #
    #     # train
    #     for id in range(split):
    #         pos = build_pos_pairs_for_imageid(id)
    #         neg = build_neg_pairs_for_imageid(id, list(range(split)))
    #         for pair in pos:
    #             listX1 += [pair[0]]
    #             listX2 += [pair[1]]
    #             listY += [1]
    #         for pair in neg:
    #             if sum(listY) > len(listY) / 2:
    #                 listX1 += [pair[0]]
    #                 listX2 += [pair[1]]
    #                 listY += [0]
    #     perm = np.random.permutation(len(listX1))
    #     X1_ids_train = np.array(listX1)[perm]
    #     X2_ids_train = np.array(listX2)[perm]
    #     Y_ids_train = np.array(listY)[perm]
    #
    #     listX1 = []
    #     listX2 = []
    #     listY = []
    #
    #     # test
    #     for id in range(split, num_classes):
    #         pos = build_pos_pairs_for_imageid(id)
    #         neg = build_neg_pairs_for_imageid(id, list(range(split, num_classes)))
    #         for pair in pos:
    #             listX1 += [pair[0]]
    #             listX2 += [pair[1]]
    #             listY += [1]
    #         for pair in neg:
    #             if sum(listY) > len(listY) / 2:
    #                 listX1 += [pair[0]]
    #                 listX2 += [pair[1]]
    #                 listY += [0]
    #     X1_ids_test = np.array(listX1)
    #     X2_ids_test = np.array(listX2)
    #     Y_ids_test = np.array(listY)
    #     return (X1_ids_train, X2_ids_train, Y_ids_train,
    #             X1_ids_test, X2_ids_test, Y_ids_test)
    #
    #
    # class Generator:
    #
    #     def __init__(self, X1, X2, Y, batch_size, all_imgs):
    #         self.cur_train_index = 0
    #         self.batch_size = batch_size
    #         self.X1 = X1
    #         self.X2 = X2
    #         self.Y = Y
    #         self.imgs = all_imgs
    #         self.num_samples = Y.shape[0]
    #
    #     def next_train(self):
    #         while 1:
    #             self.cur_train_index += self.batch_size
    #             if self.cur_train_index >= self.num_samples:
    #                 self.cur_train_index = 0
    #
    #             imgs1 = self.X1[self.cur_train_index:self.cur_train_index + self.batch_size]
    #             imgs2 = self.X2[self.cur_train_index:self.cur_train_index + self.batch_size]
    #
    #             yield ([self.imgs[imgs1], self.imgs[imgs2]],
    #                    self.Y[self.cur_train_index:self.cur_train_index + self.batch_size])
    #
    #
    # # In[52]:
    #
    #
    # # Generate training imageid index and test imageid index
    # num_classes = len(name_to_classid)  # Total number of persons used in this workshop
    # X1_ids_train, X2_ids_train, train_Y, X1_ids_test, X2_ids_test, test_Y = build_train_test_data(num_classes)
    #
    # # Open all images
    # all_imgs = open_all_images(imageid_to_path)
    # print('Total images: ', all_imgs.shape)
    #
    # # Define an image dataloader
    # gen = Generator(X1_ids_train, X2_ids_train, train_Y, 32, all_imgs)
    # [x1, x2], y = next(gen.next_train())
    #
    # # Prepare test image subset
    # test_X1 = all_imgs[X1_ids_test]
    # test_X2 = all_imgs[X2_ids_test]
    #
    #
    # # # Design and train a person verification model
    # #
    # # - Build a convolutional model. Large convolutions on high dimensional images can be very slow on CPUs.
    #
    # # In[53]:
    #
    #
    # # Define the loss function
    # def contrastive_loss(y_true, y_pred, margin=0.25):
    #     '''Contrastive loss from Hadsell-et-al.'06
    #     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    #     '''
    #     return K.mean(y_true * K.square(1 - y_pred) +
    #                   (1 - y_true) * K.square(K.maximum(y_pred - margin, 0)))
    #
    #
    # def accuracy_sim(y_true, y_pred, threshold=0.5):
    #     '''Compute classification accuracy with a fixed threshold on similarity.
    #     '''
    #     y_thresholded = K.cast(y_pred > threshold, y_true.dtype)
    #     return K.mean(K.equal(y_true, y_thresholded))
    #
    #
    # # In[54]:
    #
    #
    # # Define a model
    # inp = Input((100, 100, 3), dtype='float32')
    # x = Conv2D(16, 3, activation="relu", padding="same")(inp)
    # x = Conv2D(16, 3, activation="relu", padding="same")(x)
    # x = MaxPool2D((2, 2))(x)
    # x = Flatten()(x)
    # x = Dense(50)(x)
    # shared_conv = Model(inputs=inp, outputs=x)
    #
    # print('Input: ', all_imgs.shape)
    # print('Output: ', shared_conv.predict(all_imgs).shape)
    #
    # shared_conv.summary()
    #
    # # In[55]:
    #
    #
    # # Define the siamese model
    # i1 = Input((100, 100, 3), dtype='float32')
    # i2 = Input((100, 100, 3), dtype='float32')
    #
    # x1 = shared_conv(i1)
    # x2 = shared_conv(i2)
    #
    # out = Dot(axes=-1, normalize=True)([x1, x2])
    #
    # model = Model(inputs=[i1, i2], outputs=out)
    # model.compile(loss=contrastive_loss, optimizer='rmsprop', metrics=[accuracy_sim])
    #
    # model.summary()
    #
    # # In[56]:
    #
    # # Train the model, set smaller epoch if you don't have GPU
    #
    # # NUM_EPOCH = 20
    # # hist = model.fit_generator(generator=gen.next_train(), steps_per_epoch=train_Y.shape[0] // 32, epochs=NUM_EPOCH,
    # #                            validation_data=([test_X1, test_X2], test_Y), verbose=2)
    #
    # # TODO load the model file
    # model = tf.keras.models.load_model('model_fullTrain.h5', custom_objects={'contrastive_loss': contrastive_loss, 'accuracy_sim' : accuracy_sim})
    #
    # # Load images and resize images
    # # import random
    #
    # # img_path = random.sample(list(imageid_to_path.values()), 1)
    # img_path = cfg.img
    # img_path = "".join(img_path)
    # # print('Randomly selected sample image is: ' + img_path)
    # print('Newly captured sample image is: ' + img_path)
    # print()
    # test_img = cv2.imread(img_path)
    # test_img = cv2.resize(test_img, (100, 100))
    #
    # # Extract features using the pre-trained siamese model
    # test_img_coeff = shared_conv.predict(np.expand_dims(test_img, 0))
    #
    # # Normalize features and calculate their cosine similarity distance
    # sims_arr = []
    # for i in range(len(imageid_to_path)):
    #     test_img2_coeff = shared_conv.predict(np.expand_dims(all_imgs[i], 0))
    #     sims = np.inner(test_img_coeff / np.linalg.norm(test_img_coeff), test_img2_coeff / np.linalg.norm(test_img2_coeff))
    #     # store cosine similarity value align with imageid_to_path index
    #     sims_arr.append(sims[0, 0])
    # # print(sims_arr)
    # # Find out the most similar image
    # index1 = np.argmax(sims_arr)
    # print('The 1st high similarity is: %.4f' % sims_arr[index1])
    # print('comparing to: ' + imageid_to_path[index1])
    # print()
    # similarity = sims_arr[index1]
    # folderPath = imageid_to_path[index1]
    # # reset max to min for finding next max
    # sims_arr[index1] = 0
    #
    # index2 = np.argmax(sims_arr)
    # print('The 2nd high similarity is: %.4f' % sims_arr[index2])
    # print('comparing to: ' + imageid_to_path[index2])
    # print()
    # sims_arr[index2] = 0
    #
    # index3 = np.argmax(sims_arr)
    # print('The 3rd high similarity is: %.4f' % sims_arr[index3])
    # print('comparing to: ' + imageid_to_path[index3])
    # print()
    # sims_arr[index3] = 0
    #
    # plt.figure()
    # plt.subplot(2, 3, 1)
    # plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) / 255)
    # plt.axis('off')
    # plt.subplot(2, 3, 4)
    # plt.imshow(cv2.cvtColor(all_imgs[index1], cv2.COLOR_BGR2RGB) / 255)
    # plt.axis('off')
    # plt.subplot(2, 3, 5)
    # plt.imshow(cv2.cvtColor(all_imgs[index2], cv2.COLOR_BGR2RGB) / 255)
    # plt.axis('off')
    # plt.subplot(2, 3, 6)
    # plt.imshow(cv2.cvtColor(all_imgs[index3], cv2.COLOR_BGR2RGB) / 255)
    # plt.axis('off')
    # plt.show()
    similarity = 0.99
    folderPath = 'lfw-deepfunneled/A_Lijian_Gu/Lijian_Gu_0001.jpg'
    return similarity, folderPath

class AutoBooking(MycroftSkill):
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def __init__(self):
        MycroftSkill.__init__(self)

    @intent_file_handler('booking.auto.intent')
    def handle_booking_auto(self, message):

        camera = PiCamera()
        camera.start_preview()
        sleep(5)
        camera.capture('/home/pi/ISAPM/temp/image.jpg')
        camera.stop_preview()

        cfg = get_args()
        sim, simImg = identify_face(cfg)
        print('Similarity is: ' + str(sim) + ' folderPath is: ' + simImg)
        if sim >= 0.95:
            dirlist = simImg.split('/')
            revDirList = reversed(dirlist)
            name = (list(revDirList))[0].split('_')
            name.pop()
            firstName = name[0]
            lastName = name[1]
            print('Hello ' + firstName + ' ' + lastName + ' , welcome back!')
            self.speak_dialog('Hello ' + firstName + ' ' + lastName + ' , welcome back')


        while True:
            if not firstName:
                firstName = self.get_response("What is you first name")
            firstNameTrim = firstName.replace(" ", "")
            confirm = self.get_response("Please confirm your first name is " + firstNameTrim)
            confirmLower = confirm.lower()

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        while True:
            if not lastName:
                lastName = self.get_response("What is you last name")
            lastNameTrim = lastName.replace(" ", "")
            confirm = self.get_response("Please confirm your last name is " + lastNameTrim)
            confirmLower = confirm.lower()

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        while True:
            mobileNumber = self.get_response("What is you mobile number")
            mobileNumberTrim = mobileNumber.replace(" ", "")
            mobileNumber = mobileNumberTrim + "/"
            if (str.isdigit(mobileNumberTrim) and len(mobileNumberTrim)==8):
                confirm = self.get_response("Please confirm your mobile number is " + mobileNumber)
                confirmLower = confirm.lower()
            else:
                continue

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break
        
        while True:
            email = self.get_response("What is you email address")
            emailList = email.split(" ")
            newEmailList = ["@" if e == "at" else e for e in emailList]
            emailTrim = ''.join(newEmailList)
            confirm = self.get_response("Please confirm your email address is " + emailTrim)
            confirmLower = confirm.lower()

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        while True:
            postalCode = self.get_response("What is you current address 6-digit postal code?")
            postalCodeTrim = postalCode.replace(" ", "")
            postalCode = postalCodeTrim + "/"
            confirm = self.get_response("Please confirm your address postal code is " + postalCode)
            confirmLower = confirm.lower()

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        while True:
            dob = self.get_response("What is your birthday(YYYYMMDD)?")
            dobTrim = dob.replace(" ", "")
            dob = dobTrim + "_"
            if (str.isdigit(dobTrim) and len(dobTrim)==8):
                confirm = self.get_response("Please confirm your birthday is " + dob)
                confirmLower = confirm.lower()
            else:
                continue

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        while True:
            facility = self.get_response("Which type of facility do you want to book (Hospital or Polyclinic)?")
            confirm = self.get_response("Please confirm facility type you choose to book is " + facility)
            confirmLower = confirm.lower()

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        while True:
            bookingDate = self.get_response("Which date do you want to book (YYYYMMDD)?")
            bookingDateTrim = bookingDate.replace(" ", "")
            bookingDate = bookingDateTrim + "/"
            if (str.isdigit(bookingDateTrim) and len(bookingDateTrim)==8):
                confirm = self.get_response("Please confirm your booking date is " + bookingDate)
                confirmLower = confirm.lower()
            else:
                continue

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        while True:
            bookingTime = self.get_response("What time do you want to book (HHMM)?")
            confirm = self.get_response("Please confirm your booking time is " + bookingTime)
            confirmLower = confirm.lower()

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        self.log.info("firstName:" + firstNameTrim + ",lastName:" + lastNameTrim + ", contactNumber:" + mobileNumberTrim +
                      ", email:" + emailTrim + ", dob:" + dobTrim + ", facility:" + facility + ", bookingDate:" + bookingDateTrim)

        url = 'http://18c24f0c10aa.ngrok.io/bookingsystem'
        myobj = {
          "First_Name": firstNameTrim,
          "Last_Name": lastNameTrim,
          "Contact_No": mobileNumberTrim,
          "Email": emailTrim,
          "Address": postalCodeTrim,
          "DOB": dobTrim,
          "Medical_Description" : "",
          "Treatment_Facility" : facility,
          "Booking_Timing": bookingTime,
          "Booking_Date": bookingDateTrim,
          "Sore_Throat": "true",
          "Fever": "false"
        }

        # while True:
        #     try:
        #         sleep(5)
        #         bucket = client.bucket("e-charger-303306.appspot.com")
        #         x = datetime.datetime.now()
        #         blob = bucket.blob(firstNameTrim + lastNameTrim + "-" + x.strftime("%m") + "-" + x.strftime("%d") + "-" + x.strftime("%y"))
        #         url = blob.generate_signed_url(version="v4", expiration=datetime.timedelta(minutes=15), method="GET")
        #         webbrowser.open(url)
        #         self.speak_dialog('Please scan the qr code to continue')
        #         break
        #     except:
        #         continue

        res = requests.post(url, json = myobj)
        self.log.info(res)

        if res.result:
            self.speak_dialog('Your booking is successefully')
        else:
            self.speak_dialog('Sorry, your booking is failed')


def create_skill():
    return AutoBooking()

