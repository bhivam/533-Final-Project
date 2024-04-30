import json
import numpy


filtered_file = json.load(open("filtered_file.json", "r"))
annotated_file = json.load(open("label_word.json", "r"))
variation_threshold = numpy.percentile(filtered_file["variation_rate"], 90)
count = 0
temp_data = {}
alignment_list = []
for i in annotated_file.keys():
    if annotated_file[i]["variation_rate"] >= variation_threshold:
        temp_data[i] = {"original":annotated_file[i]["original"], "text_to_keep": annotated_file[i]["text_to_keep"], "labels": annotated_file[i]["labels"], "alignment_gap": annotated_file[i]["alignment_gap"]}
        alignment_list.append(annotated_file[i]["alignment_gap"])

final_data = {}
alignment_threshold = numpy.percentile(alignment_list, 90)
for i in temp_data.keys():
    if temp_data[i]["alignment_gap"] >= alignment_threshold:
        final_data[count] = {"original":temp_data[i]["original"], "text_to_keep": temp_data[i]["text_to_keep"], "labels": temp_data[i]["labels"]}
        count = count + 1

json.dump(final_data, open("quality_control_data.json", "w"), indent=4)