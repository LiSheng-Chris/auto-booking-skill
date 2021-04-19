from mycroft import MycroftSkill, intent_file_handler
from picamera import PiCamera
from time import sleep


class AutoBooking(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self)

    @intent_file_handler('booking.auto.intent')
    def handle_booking_auto(self, message):
        camera = PiCamera()
        camera.start_preview()
        sleep(3)
        camera.capture('/home/pi/ISAPM/temp/image.jpg')
        camera.stop_preview()
        bookingDate = self.get_response('Which date you want to booking')
        self.speak_dialog(bookingDate)

def create_skill():
    return AutoBooking()

