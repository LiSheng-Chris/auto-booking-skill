from mycroft import MycroftSkill, intent_file_handler
# from picamera import PiCamera
# from time import sleep
import requests
import json
import webbrowser

class AutoBooking(MycroftSkill):
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def __init__(self):
        MycroftSkill.__init__(self)

    @intent_file_handler('booking.auto.intent')
    def handle_booking_auto(self, message):
        ## Lijian part start
        # self.speak_dialog("init camera")
        # camera = PiCamera()
        # self.speak_dialog("starting preview")
        # camera.start_preview()
        # self.speak_dialog("wait for 5 seconds")
        # sleep(5)
        # path = '/home/pi/ISAPM/temp/image.jpg'
        # self.speak_dialog("taking photo capture")
        # camera.capture(path)
        # camera.stop_preview()
        # self.speak_dialog("taking photo succeeded!")
        ## Lijian part end
        self.log.info("Open browser")
        webbrowser.open("https://firebasestorage.googleapis.com/v0/b/e-charger-303306.appspot.com/o/speed-internet-technology-background.jpg?alt=media&token=63244996-e359-48f1-b3b9-0b73f73554f2")

        # camera = PiCamera()
        # camera.start_preview()
        # sleep(5)
        # camera.capture('/home/pi/ISAPM/temp/image.jpg')
        # camera.stop_preview()

        ## Li Sheng part start
        self.speak_dialog("Hi, prepare to show new image.")
        self.gui.clear()
        self.enclosure.display_manager.remove_active()
        self.gui.show_image("https://source.unsplash.com/1920x1080/?+random", "Example Long Caption That Needs Wrapping Very Long Long Text Text Example That Is", "Example Title", "PreserveAspectFit", 10)
        self.gui.show_text('gui.show_text testing')
        # self.gui.show_image("https://placeimg.com/500/300/nature")
        # self.speak_dialog("Hi, show image is finished!")

        while True:
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
            contactNumber = self.get_response("What is you mobile number")
            contactNumberTrim = contactNumber.replace(" ", "")
            if (str.isdigit(contactNumberTrim) and len(contactNumberTrim)==8):
                confirm = self.get_response("Please confirm your contact number is " + contactNumberTrim)
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
            address = self.get_response("What is you current address postal code")
            addressTrim = contactNumber.replace(" ", "")
            confirm = self.get_response("Please confirm your address postal code is " + addressTrim)
            confirmLower = confirm.lower()

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        while True:
            dob = self.get_response("What is your birthday? YYYY-MM-DD")
            confirm = self.get_response("Please confirm your birthday is " + dob)
            confirmLower = confirm.lower()

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        while True:
            facility = self.get_response("Which type of facility do you prefer? Hospital or Polyclinic")
            confirm = self.get_response("Please confirm the type of facility you choose is " + facility)
            confirmLower = confirm.lower()

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        while True:
            bookingDate = self.get_response("Which date you want to book (YYYY-MM-DD)")
            confirm = self.get_response("Please confirm your booking date is " + bookingDate)
            confirmLower = confirm.lower()

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        while True:
            bookingTime = self.get_response("What time do you want to book? HH-MM")
            confirm = self.get_response("Please confirm your booking time is " + bookingTime)
            confirmLower = confirm.lower()

            if "stop" in confirmLower:
                return
            elif "no" in confirmLower or "incorrect" in confirmLower or "wrong" in confirmLower:
                continue
            else:
                break

        # firstName = self.get_response("What is you first name")
        # lastName = self.get_response("What is you last name")
        # self.speak_dialog("Hi " + firstName + " " + lastName)
        # contactNumber = self.get_response("What is you contact number")
        # self.speak_dialog("Your contact number is " + contactNumber)
        # email = self.get_response("What is you email address")
        # self.speak_dialog("Your email address is " + email)
        # address = self.get_response("Where is you current location")
        # self.speak_dialog("Your location is " + address)
        # dob = self.get_response("When is your birthday")
        # self.speak_dialog("Your birthday is " + dob)
        # facility = self.get_response("Which type of facility you prefer")
        # self.speak_dialog("The type of facility you choose is " + facility)
        # bookingDate = self.get_response("Which date you want to booking")
        # self.speak_dialog("Your booking date is " + bookingDate)
        # bookingTime = self.get_response("What time do you prefer")
        # self.speak_dialog("Your booking time is " + bookingTime)
        self.log.info("firstName:" + firstName + ",lastName:" + lastName + ", contactNumber:" + contactNumber + ", email:" + email + ", dob:" + dob + ", facility:" + facility + ", bookingDate:" + bookingDate)

        url = 'http://8d9fb6d6e740.ngrok.io/bookingsystem'
        myobj = {
          "First_Name": firstNameTrim,
          "Last_Name": lastNameTrim,
          "Contact_No": contactNumberTrim,
          "Email": emailTrim,
          "Address": addressTrim,
          "DOB": dob,
          "Medical_Description" : "",
          "Treatment_Facility" : facility,
          "Booking_Timing": bookingTime,
          "Booking_Date": bookingDate,
          "Sore_Throat": "true",
          "Fever": "false"
        }

        res = requests.post(url, json = myobj)
        self.log.info(res)
        ## Li Sheng part end
        
        ## Yan Bo part start
        ## Yan Bo part end

def create_skill():
    return AutoBooking()

