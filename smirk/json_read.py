import json

json_dict=json.load(open("./output.json","r"))

print(list(json_dict.keys())[0])

print(json_dict[list(json_dict.keys())[0]]["clip_data_list"][0])
