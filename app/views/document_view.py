from abc import ABC, abstractmethod


class DocumentView(ABC):
    has_item_view = False

    # def __init__(self, document: Document = None):
    #     self.document = document
    #     self.has_item_view = False

    @property
    @abstractmethod
    def fields(self):
        pass
