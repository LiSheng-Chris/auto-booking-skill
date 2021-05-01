from mycroft import MycroftSkill, intent_file_handler
from picamera import PiCamera
from time import sleep
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

        camera = PiCamera()
        camera.start_preview()
        sleep(5)
        camera.capture('/home/pi/ISAPM/temp/image.jpg')
        camera.stop_preview()

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


def create_skill():
    return AutoBooking()

