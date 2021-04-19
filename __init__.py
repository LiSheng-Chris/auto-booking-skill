from mycroft import MycroftSkill, intent_file_handler


class AutoBooking(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self)

    @intent_file_handler('booking.auto.intent')
    def handle_booking_auto(self, message):
        bookingDate = self.get_response('Which date you want to booking')
        self.speak_dialog(bookingDate)

def create_skill():
    return AutoBooking()

