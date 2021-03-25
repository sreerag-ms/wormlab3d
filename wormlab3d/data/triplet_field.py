from mongoengine import ListField


class TripletField(ListField):
    def validate(self, value):
        if len(value) != 0 and len(value) != 3:
            self.error('TripletField must contain exactly 3 values.')
        super().validate(value)
