import json

def delete_key_inside_annotations(input_file, output_file, key_to_delete):
    with open(input_file, 'r') as f:
        data = json.load(f)
        #print(data["annotations"])

    for i in range(len(data["annotations"])):
        #print(type(data["annotations"][i]))
        if key_to_delete in data["annotations"][i]:
            # Delete the specified key from "annotations"
            del data["annotations"][i]["segmentation"]
    
    # Write the modified JSON data to a new file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage:
input_file = 'output.json'  # Path to the input JSON file
output_file = 'output.json'  # Path to the output JSON file
key_name = 'annotations'  # Key name to check
key_to_delete = "segmentation" 

delete_key_inside_annotations(input_file, output_file, key_to_delete)