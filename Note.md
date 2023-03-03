# CSVDataset

- all = json.load('imsitu_space.json')
- verb_info = all['verbs']
```json
"drinking": {
"framenet": "Ingestion", 
"abstract": "the AGENT drinks a LIQUID from a CONTAINER at a PLACE", 
"def": "take (a liquid) into the mouth and swallow", 
"order": ["agent", "liquid", "container", "place"], 
"roles": {"place": {"framenet": "place", "def": "The location where the drink event is happening"}, 
        "container": {"framenet": "source", "def": "The container in which the liquid is in"}, 
        "liquid": {"framenet": "ingestibles", "def": "The entity that the agent is drinking"}, 
        "agent": {"framenet": "ingestor", "def": "The entity doing the drink action"}}
}
```
- verb_role = {verb: value['order'] for verb, value in verb_info.items()}
- vidx_ridx: convert role in verb_role to role_index
- classes, idx_to_class: from train_classes.csv
- verb_to_idx, idx_to_verb: from verb_indices.txt
- role_to_idx, idx_to_role: from role_indices.txt
- SWiG_json: from train.json
- image_data = self._read_annotations(self.SWiG_json, verb_info, self.classes)
```
Load annotations for a image, if roles is less than six, padding to six.
[{'x1': 101, 'x2': 585, 'y1': 130, 'y2': 382, 'class1': 'n01503061', 'class2': 'n01503061', 'class3': 'n01503061'}, 
{'x1': 99, 'x2': 596, 'y1': 92, 'y2': 345, 'class1': 'n02151625', 'class2': 'n04592962', 'class3': 'n00179916'}, 
{'x1': -1, 'x2': -1, 'y1': -1, 'y2': -1, 'class1': 'n08613733', 'class2': 'n09328904', 'class3': 'n09436708'}, 
{'x1': -1, 'x2': -1, 'y1': -1, 'y2': -1, 'class1': 'Pad', 'class2': 'Pad', 'class3': 'Pad'}, 
{'x1': -1, 'x2': -1, 'y1': -1, 'y2': -1, 'class1': 'Pad', 'class2': 'Pad', 'class3': 'Pad'}, 
{'x1': -1, 'x2': -1, 'y1': -1, 'y2': -1, 'class1': 'Pad', 'class2': 'Pad', 'class3': 'Pad'}]
```
- image_names = list(self.image_data.keys())
- image_to_image_idx: index in image_names

