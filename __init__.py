from mycroft import MycroftSkill, intent_file_handler


class AutoBooking(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self)

    @intent_file_handler('booking.auto.intent')
    def handle_booking_auto(self, message):
        self.speak_dialog('booking.auto')


def create_skill():
    return AutoBooking()

