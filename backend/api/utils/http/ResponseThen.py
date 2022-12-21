from django.http import HttpResponse


class ResponseThen(HttpResponse):
    """
    A subclass of HttpResponse that calls a callback when closed.
    It allows to send a response to the client and then perform some actions.
    """
    def __init__(self, response, callback):
        super().__init__(response)
        self.callback = callback

    def close(self):
        super().close()
        self.callback()