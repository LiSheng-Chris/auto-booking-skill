from mycroft import MycroftSkill, intent_file_handler


class AutoBooking(MycroftSkill):
    def __init__(self):
        MycroftSkill.__init__(self)

    @intent_file_handler('booking.auto.intent')
    def handle_booking_auto(self, message):
        favorite_flavor = self.get_response('booking.auto.dialog')
        self.speak_dialog('confirm.favorite.flavor', {'flavor': favorite_flavor})


def create_skill():
    return AutoBooking()

