import json

json_data = '''
[
  {"Dispatcher": "Thank you for using Live Safe...What is your location please?"},
  {"User": "### Blackstone apt ##"}
]
'''

parsed_data = json.loads(json_data)
print(parsed_data)
