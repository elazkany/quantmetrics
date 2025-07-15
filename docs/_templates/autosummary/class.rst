{{ fullname.split('.')[-1] }}
{{ "=" * (fullname.split('.')[-1]|length) }}

.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index: