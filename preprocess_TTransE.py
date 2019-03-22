import numpy as np

time_dict = {}

path = "data/ICEWS18_original/"
newpath = "data/ICEWS18_TTransE/"

fr_stat = open(path + "stat.txt", "r")
fw_stat = open(newpath + "stat.txt", "w")

count = 0
time_index = 3
entity_total, relation_total, _ = fr_stat.readline().split()

def preprocess(data_part):
    data_path = path+data_part+"2id.txt"
    write_path = newpath+data_part+"2id.txt"
    fw = open(write_path, "w")
    with open(data_path) as fp:
        for i,line in enumerate(fp):
            global count
            count += 1
            info = line.strip().split("\t")

            time_id = None
            if info[time_index] in time_dict:
                time_id = time_dict[info[time_index]]
            else:
                time_id = len(time_dict)
                time_dict[info[time_index]] = time_id

            fw.write("%-8d %-8d %-8d %-8d\n" % (int(info[0]), int(info[1]), int(info[2]), time_id))

    fw.close()

preprocess("train")
preprocess("valid")
preprocess("test")
print(count)
fw_stat.write(entity_total + " " + relation_total + " " + str(len(time_dict)))
fw_stat.close()
