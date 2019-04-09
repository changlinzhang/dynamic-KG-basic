import datetime

time_dict = {}

# path = "data/ICEWS18_original/"
# newpath = "data/ICEWS18_TTransE/"
path = "data/GDELT_original/"
newpath = "data/GDELT_TTransE/"

fw = open(newpath + "timedict.txt", "w")

count = 0
time_index = 3

start_time_str = "2018-01-01-00:00"
format = "%Y-%m-%d-%H:%M"
start_time = datetime.datetime.strptime(start_time_str, format)

preTimestamp = -1

def preprocess(data_part):
    data_path = path+data_part+"2id.txt"
    with open(data_path) as fp:
        for i,line in enumerate(fp):
            global count
            count += 1
            info = line.strip().split("\t")

            if not info[time_index] in time_dict:
                time_id = len(time_dict)
                time_dict[info[time_index]] = time_id
                time = start_time + datetime.timedelta(minutes=int(info[time_index]))
                time_str = time.strftime(format)
                fw.write("%d %s\n" % (time_id, time_str))

preprocess("train")
preprocess("valid")
preprocess("test")
print(count)
fw.close()
