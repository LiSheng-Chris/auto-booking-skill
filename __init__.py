from mycroft import MycroftSkill, intent_file_handler
from picamera import PiCamera
from time import sleep


class AutoBooking(MycroftSkill):
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
        
        ## Li Sheng part start
        self.log.info("Main method started！")
        name = self.get_response("What is you name")
        self.speak_dialog("Hi " + name)
        nric = self.get_response("What is you NRIC number")
        self.speak_dialog("Your nric number is " + nric)
        email = self.get_response("What is you email address")
        self.speak_dialog("Your email address is " + email)
        bookingDate = self.get_response("Which date you want to booking")
        self.speak_dialog("Your booking date is " + bookingDate)
        self.log.info("name:" + name + ", nric:" + nric + ", email:" + email + ", bookingDate:" + bookingDate)
        ## Li Sheng part end
        
        ## Yan Bo part start
        ## Yan Bo part end

def create_skill():
    return AutoBooking()

