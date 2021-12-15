from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Dict, Any

from flask_wtf import FlaskForm
from mongoengine import Document
from wtforms import IntegerField, FloatField, StringField, SelectField, FormField


NESTED_DOCUMENT_SEPARATOR = '__'


class DocumentView(ABC):
    has_item_view = False

    def __init__(self, document: Document = None, hide_fields: list = None, field_values: list = None, prefix: str=''):
        self.document = document

        # Add prefix
        if prefix != '':
            prefix = prefix + NESTED_DOCUMENT_SEPARATOR
        self.prefix = prefix

        # Initialise the field specifications
        self.fields = self._init_fields()

        # Hide fields - expanding any wildcards
        if hide_fields is None:
            hide_fields = []
        for hk in hide_fields:
            if hk[-1] != '*':
                continue
            for key in self.fields.keys():
                if key[:len(hk)-1] == hk[:-1]:
                    hide_fields.append(key)
        self.hide_fields = hide_fields

        # Set initial field values
        if field_values is None:
            field_values = []
        self.field_values = field_values

    @classmethod
    @property
    @abstractmethod
    def document_class(cls) -> Document:
        pass

    @classmethod
    @property
    def collection_name(cls) -> str:
        return cls.document_class._get_collection_name()

    @abstractmethod
    def _init_fields(self) -> OrderedDict[str, Dict[str, str]]:
        pass

    def get_choices(self, key: str) -> List[Any]:
        if NESTED_DOCUMENT_SEPARATOR in key:
            rel_collection, rel_key = key.split('__')
            return self.fields[rel_collection]['view_class'].get_choices(rel_key)
        return self.document_class.objects.distinct(key)

    # @classmethod
    def filters_form(self, prefix=None, prefix_idx=None, instantiate=True) -> FlaskForm:
        class FilterForm(FlaskForm):
            pass
        # print('\n')
        # print('FilterForm', prefix)
        # print(self.fields)

        if prefix_idx is None:
            prefix_idx = ''
        else:
            prefix_idx = prefix_idx + '.'

        for i, (key, field) in enumerate(self.fields.items()):


            label = field['title']
            filter_type = field['filter_type'] if 'filter_type' in field else field['type']

            if filter_type == 'none':
                continue

            if key in self.hide_fields:
                continue

            render_kw = {'class': 'form-control', 'data-idx': prefix_idx + str(i)}
            form_field = None
            if filter_type == 'integer':
                form_field = IntegerField(label, render_kw=render_kw)
            elif filter_type == 'float':
                form_field = FloatField(label, render_kw=render_kw)
            elif filter_type == 'string':
                form_field = StringField(label, render_kw=render_kw)
            elif filter_type == 'choice_query':
                values = self.get_choices(key)
                choices = [''] + values
                render_kw['class'] += ' form-select'
                form_field = SelectField(label, choices=choices, render_kw=render_kw)

            if form_field is not None:
                setattr(FilterForm, key, form_field)

            # if field['type'] == 'relation':
            #     rel_form = field['view_class'].filters_form(instantiate=False, prefix_idx=prefix_idx+str(i))
            #     setattr(FilterForm, 'rel_' + field['key'], FormField(rel_form))

        if not instantiate:
            return FilterForm

        if prefix is None:
            prefix = 'filters_'
        # print(prefix, FilterForm)
        form = FilterForm(prefix=prefix)
        # for field in form:
        #     print(field)
        # print('\n')

        return form
        # return FilterForm(prefix=prefix)
